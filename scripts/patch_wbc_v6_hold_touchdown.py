from pathlib import Path

p = Path("envs/g1_wbc_taskspace_right_lift_env.py")
s = p.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1) RECOVERY must keep touchdown force active until land_end.
# ------------------------------------------------------------
old = '''        if self._wbc_state == "RECOVERY":
            self._touchdown_timer += self.dt
            self._touchdown_force = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))
            if self._touchdown_timer >= 0.30:
                self._wbc_state = "SETTLE"
            return
'''

new = '''        if self._wbc_state == "RECOVERY":
            self._touchdown_timer += self.dt

            # V6 fix:
            # Touchdown succeeded briefly, but v5 released control while the
            # parent swing phase was still active. Keep hard touchdown ON until
            # the original landing phase is over.
            self._touchdown_force = 1.0

            if right_contact and phi >= float(self.cfg.land_end):
                self._wbc_state = "SETTLE"
                self._touchdown_force = 0.0
            return
'''

if old not in s:
    raise RuntimeError("Could not find RECOVERY decay block. File differs from expected v5.")
s = s.replace(old, new, 1)

# ------------------------------------------------------------
# 2) Remove aggressive recovery brake push.
#    It was unloading the left support foot.
# ------------------------------------------------------------
old = '''        # WBC v5 RECOVERY: both-feet braking after right_contact.
        if self._wbc_state == "RECOVERY":
            rec = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))
            target[0] += -0.06 * rec
            target[4] += -0.08 * rec
            target[6] += -0.06 * rec
            target[10] += -0.08 * rec
            target[14] += +0.05 * rec

'''

new = '''        # WBC v6 RECOVERY:
        # Do not add extra brake push here. The hard touchdown posture already
        # keeps the right foot down. Extra push caused left support unload/hop.
        if self._wbc_state == "RECOVERY":
            pass

'''

if old not in s:
    raise RuntimeError("Could not find RECOVERY brake block.")
s = s.replace(old, new, 1)

# ------------------------------------------------------------
# 3) Make support-side press gentler during touchdown.
# ------------------------------------------------------------
s = s.replace("target[3] += +0.06 * land      # left knee slight flexion/press",
              "target[3] += +0.015 * land     # gentle left knee press")
s = s.replace("target[0] += -0.05 * land      # left hip extension",
              "target[0] += -0.015 * land     # gentle left hip extension")
s = s.replace("target[4] += -0.06 * land      # left ankle plantarflexion",
              "target[4] += -0.015 * land     # gentle left ankle plantarflexion")
s = s.replace("target[14] += +0.04 * land     # forward trunk during touchdown",
              "target[14] += +0.010 * land    # gentle forward trunk during touchdown")

# ------------------------------------------------------------
# 4) Reward version marker.
# ------------------------------------------------------------
s = s.replace(
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v5_hard_state_touchdown"',
    'rinfo["reward_version"] = "wbc_taskspace_right_lift_v6_hold_touchdown_until_land_end"'
)

p.write_text(s, encoding="utf-8")
print("WBC v6 hold-touchdown patch applied.")