from pathlib import Path

p = Path("envs/g1_wbc_taskspace_right_lift_env.py")
s = p.read_text(encoding="utf-8")

if "wbc_taskspace_right_lift_v5_hard_state_touchdown" in s:
    print("WBC v5 patch already appears to be applied.")
    raise SystemExit(0)

# ------------------------------------------------------------
# 1) Add state-machine variables in __init__ and reset
# ------------------------------------------------------------
needle = "        self._abort_count = 0\n"
addition = """        self._wbc_state = "PRELOAD"
        self._guard_count = 0
        self._touchdown_anchor_x = 0.0
        self._touchdown_timer = 0.0
        self._touchdown_force = 0.0
"""

if "self._wbc_state" not in s:
    s = s.replace(needle, needle + addition)

# ------------------------------------------------------------
# 2) Insert hard state-machine method
# ------------------------------------------------------------
marker = "    # ---------------------------- capture foot placement ----------------------------\n"

state_method = r'''    # ---------------------------- hard state machine ----------------------------

    def _update_wbc_state(self, info: Dict[str, float]) -> None:
        """
        WBC v5 hard state machine.

        PRELOAD/LIFT/HOLD are nominal.
        TOUCHDOWN is a hard takeover:
        - parent lift is no longer allowed to keep the foot high
        - right knee is extended
        - right ankle is plantarflexed
        - foot target is anchored to CURRENT left support foot
        """
        phi = float(info["phase"])
        sw = float(info["swing_env"])
        x = float(info["x_position"])
        xv = float(info["x_velocity"])
        up_z = float(info["up_z"])

        left_contact = bool(info["left_contact"])
        right_contact = bool(info["right_contact"])

        # Nominal phase label.
        if self._wbc_state not in ("TOUCHDOWN", "RECOVERY", "SETTLE"):
            if phi < float(self.cfg.swing_start):
                self._wbc_state = "PRELOAD"
            elif sw < 0.95:
                self._wbc_state = "LIFT"
            else:
                self._wbc_state = "HOLD"

        # Once touchdown starts, keep it active until right contact.
        if self._wbc_state == "TOUCHDOWN":
            self._touchdown_timer += self.dt
            self._touchdown_force = float(np.clip(self._touchdown_force + 0.16, 0.0, 1.0))

            if right_contact:
                self._wbc_state = "RECOVERY"
                self._touchdown_timer = 0.0
                self._touchdown_force = 1.0
            return

        if self._wbc_state == "RECOVERY":
            self._touchdown_timer += self.dt
            self._touchdown_force = float(np.clip(1.0 - self._touchdown_timer self._touchdown_force = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))
            if self._touchdown_timer >= 0.30:
                self._wbc_state = "SETTLE"
            return

        if self._wbc_state == "SETTLE":
            self._touchdown_force = 0.0
            return

        # Capture guard: trigger EARLY, not near xv=-0.98.
        # Use 3 consecutive control steps to avoid noise.
        guard = False

        # Momentum guard.
        if (not right_contact or sw > 0.35) and xv < -0.30:
            guard = True

        # Position guard.
        if (not right_contact or sw > 0.35) and x < -0.080:
            guard = True

        # Collapse guard.
        if up_z < 0.88 and sw > 0.25:
            guard = True

        # Support-foot unload guard: do not keep lifting if support foot hops.
        if (not left_contact) and sw > 0.15:
            guard = True

        if guard:
            self._guard_count += 1
        else:
            self._guard_count = max(0, self._guard_count - 1)

        if self._guard_count >= 3:
            self._wbc_state = "TOUCHDOWN"
            self._capture_active = True
            self._abort_lift = True
            if self._capture_count == 0:
                self._capture_count = 1

            # Anchor once, in CURRENT support-foot frame.
            # This fixes the old bug where foot returned to a stale/start frame.
            self._touchdown_anchor_x = float(info["left_foot_x"] + 0.010)
            self._touchdown_timer = 0.0
            self._touchdown_force = 0.20

'''

if state_method not in s:
    if marker not in s:
        raise RuntimeError("Could not find capture foot placement marker.")
    s = s.replace(marker, state_method + marker, 1)

# ------------------------------------------------------------
# 3) Replace the old safety update block in step()
# ------------------------------------------------------------
start = s.find("        # Safety state update before target generation.")
end = s.find("        target = self._target_joint_position(action, info)", start)

if start == -1 or end == -1:
    raise RuntimeError("Could not find old safety update block in step().")

new_step_block = """        # WBC v5 hard state-machine update.
        self._update_wbc_state(info)
        info = self._get_info()

"""
s = s[:start] + new_step_block + s[end:]

# ------------------------------------------------------------
# 4) Hard foot target during TOUCHDOWN
# ------------------------------------------------------------
old_return = """        if self._abort_lift:
            target[2] = float(self.right_foot_p0[2])

        return target
"""

new_return = """        # WBC v5 hard touchdown foot target.
        if self._wbc_state in ("TOUCHDOWN", "RECOVERY"):
            land = float(np.clip(self._touchdown_force, 0.0, 1.0))

            # Anchor to current support-foot frame captured at trigger.
            target[0] = (1.0 - land) * float(target[0]) + land * float(self._touchdown_anchor_x)

            # Force touchdown; do not allow parent lift to keep z high.
            ground_z = float(self.right_foot_p0[2] - 0.020)
            target[2] = (1.0 - land) * float(target[2]) + land * ground_z

        if self._abort_lift:
            target[2] = min(float(target[2]), float(self.right_foot_p0[2]))

        return target
"""

if old_return not in s:
    raise RuntimeError("Could not find _target_foot_position return block.")
s = s.replace(old_return, new_return, 1)

# ------------------------------------------------------------
# 5) Hard joint-level touchdown takeover
# ------------------------------------------------------------
hard_marker = "        # Anti-hop caps: keep support leg from over-pushing itself off the floor.\n"

hard_landing_block = """        # WBC v5 hard state-machine TOUCHDOWN takeover.
        # During TOUCHDOWN, parent lift must lose authority completely.
        if self._wbc_state in ("TOUCHDOWN", "RECOVERY"):
            land = float(np.clip(self._touchdown_force, 0.0, 1.0))
            pelvis_x = float(self.data.qpos[0])

            # Desired right-foot x in current support-foot frame.
            dx = float((self._touchdown_anchor_x - pelvis_x) / 0.36)
            hip_capture = float(np.clip(dx, -0.25, 0.35))

            # Absolute touchdown posture from the control analysis:
            # knee extension is the critical part that actually drops the foot.
            target[6] = (1.0 - land) * target[6] + land * hip_capture
            target[7] = (1.0 - land) * target[7] + land * self.stand_joint_pos[7]
            target[8] = (1.0 - land) * target[8] + land * self.stand_joint_pos[8]
            target[9] = (1.0 - land) * target[9] + land * 0.030
            target[10] = (1.0 - land) * target[10] + land * (-0.120)
            target[11] = (1.0 - land) * target[11] + land * self.stand_joint_pos[11]

            # Soft pelvis/support press using position-control joints.
            # This replaces torque-level downward force.
            target[3] += +0.06 * land      # left knee slight flexion/press
            target[0] += -0.05 * land      # left hip extension
            target[4] += -0.06 * land      # left ankle plantarflexion
            target[14] += +0.04 * land     # forward trunk during touchdown

        # WBC v5 RECOVERY: both-feet braking after right_contact.
        if self._wbc_state == "RECOVERY":
            rec = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))
            target[0] += -0.06 * rec
            target[4] += -0.08 * rec
            target[6] += -0.06 * rec
            target[10] += -0.08 * rec
            target[14] += +0.05 * rec

"""

if hard_marker not in s:
    raise RuntimeError("Could not find anti-hop marker.")
s = s.replace(hard_marker, hard_landing_block + hard_marker, 1)

# ------------------------------------------------------------
# 6) Add state info to logs
# ------------------------------------------------------------
info_marker = """        info["wbc_x_integral"] = float(self._x_int)
        return info
"""

info_repl = """        info["wbc_x_integral"] = float(self._x_int)
        info["wbc_state"] = str(self._wbc_state)
        info["wbc_guard_count"] = float(self._guard_count)
        info["wbc_touchdown_anchor_x"] = float(self._touchdown_anchor_x)
        info["wbc_touchdown_timer"] = float(self._touchdown_timer)
        info["wbc_touchdown_force"] = float(self._touchdown_force)
        return info
"""

if info_marker not in s:
    raise RuntimeError("Could not find _get_info insertion marker.")
s = s.replace(info_marker, info_repl, 1)

# ------------------------------------------------------------
# 7) Reward version
# ------------------------------------------------------------
s = s.replace(
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v3_capture_project"',
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v5_hard_state_touchdown"'
)

p.write_text(s, encoding="utf-8")
print("WBC v5 hard state-machine touchdown patch applied.")
print("Backup saved: envs/BACKUP_g1_wbc_taskspace_right_lift_env_before_v5_state_machine.py")
