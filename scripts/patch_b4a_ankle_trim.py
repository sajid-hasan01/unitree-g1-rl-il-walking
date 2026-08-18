"""
State-aware install of the V17-B4A ankle-only orientation trim.

Single mechanism changed from the trusted B2:

    1. ADD   : small, swing-envelope-gated, fixed ankle pitch/roll trim
               applied as an additive joint offset to
               left_ankle_pitch (target[4]) and left_ankle_roll (target[5]).
    2. ADD   : foot-orientation + trim diagnostics (info keys).
    3. ADD   : backwards-compatible evaluator columns + --ankle_trim_fraction.
    4. NO B3 : full 6D pose IK stays out (abort if _ik_pose_delta is present).
    5. NO    : changes to support gate / touchdown / recovery / COM hold.

The working env file is patched in place. The trusted checkpoint
envs/CHECKPOINT_V17_B2_STABLE_23p1mm.py is never written and remains the
byte-exact B2 recovery source.

Re-running this script is idempotent (exits 0 with "already patched").

Usage:
    python scripts/patch_b4a_ankle_trim.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "envs" / "g1_deterministic_left_lift_env.py"
EVAL_PATH = ROOT / "scripts" / "evaluate_deterministic_left_lift.py"
CHECKPOINT_PATH = ROOT / "envs" / "CHECKPOINT_V17_B2_STABLE_23p1mm.py"

ENV_MARKER = "# V17-B4A ANKLE-ONLY ORIENTATION TRIM"
EVAL_MARKER = "ankle_trim_fraction"
B3_MARKER = "_ik_pose_delta"

B2_SIGNATURES = [
    "hold_end: float = 0.54",
    "lower_end: float = 0.64",
    "swing_guard_com_start: float = 0.025",
    "swing_guard_com_stop: float = 0.070",
    "blend = (\n                    0.40\n                    + 0.60 * sw\n                )",
    "swing_z_weight: float = 1.00",
    "support_lock_weight: float = 0.78",
]


def fail(message: str) -> None:
    print(f"ABORT: {message}")
    sys.exit(1)


def insert_after(lines, anchor, block):
    for i, line in enumerate(lines):
        if anchor in line:
            return lines[: i + 1] + block + lines[i + 1 :], i
    fail(f"anchor not found: {anchor!r}")


def insert_before(lines, anchor, block):
    for i, line in enumerate(lines):
        if anchor in line:
            return lines[:i] + block + lines[i:], i
    fail(f"anchor not found: {anchor!r}")


def check_signatures(text: str) -> None:
    missing = [s for s in B2_SIGNATURES if s not in text]
    if missing:
        fail(
            "working env is NOT the trusted B2 file; missing signatures: "
            + "; ".join(repr(m) for m in missing)
        )


def py_compile(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        fail(f"py_compile failed: {path.name}")


# ----------------------------------------------------------------------
# Block definitions (inserted as literal text, line-based anchors only)
# ----------------------------------------------------------------------

CONFIG_BLOCK = [
    "\n",
    "    # ----------------------------------------------------------\n",
    "    # V17-B4A ankle-only orientation trim.\n",
    "    #\n",
    "    # Measured at the B2 peak (step ~283):\n",
    "    #   left foot world pitch ~ +9.3 deg, roll ~ +4.4 deg\n",
    "    #   full flattening correction (DLS, ankle joint space):\n",
    "    #     left_ankle_pitch ~ -9.272 deg\n",
    "    #     left_ankle_roll  ~ -4.467 deg\n",
    "    #\n",
    "    # B4A applies only a fixed FRACTION of that correction (0.20 ->\n",
    "    # pitch -1.854 deg, roll -0.893 deg), hard-clamped so a careless\n",
    "    # B4B fraction bump cannot exceed +/-2 deg pitch / +/-1 deg roll.\n",
    "    # Gated on swing_env (fade-in 0.60 -> 0.85); swing_env is forced\n",
    "    # to zero by the touchdown latch and by emergency landing, so the\n",
    "    # trim can never reach the descent / touchdown / recovery phases.\n",
    "    # ----------------------------------------------------------\n",
    "    ankle_trim_enabled: bool = True\n",
    "    ankle_trim_fraction: float = 0.20\n",
    "    ankle_trim_pitch_rad: float = math.radians(9.272)\n",
    "    ankle_trim_roll_rad: float = math.radians(4.467)\n",
    "    ankle_trim_pitch_max: float = math.radians(2.0)\n",
    "    ankle_trim_roll_max: float = math.radians(1.0)\n",
    "    ankle_trim_fade_start: float = 0.60\n",
    "    ankle_trim_fade_end: float = 0.85\n",
]

HELPERS_BLOCK = [
    "\n",
    "    # ------------------------------------------------------------------\n",
    "    # V17-B4A helpers: foot-orientation diagnostics + ankle trim law.\n",
    "    # ------------------------------------------------------------------\n",
    "\n",
    "    def _left_foot_orientation_diag(self):\n",
    "        \"\"\"\n",
    "        Foot-tilt diagnostics relative to world +Z (flat floor).\n",
    "\n",
    "        Uses only the foot body's local up axis so the decomposition is\n",
    "        decoupled from yaw:\n",
    "\n",
    "            pitch_deg: tilt of the local-Z axis about world Y\n",
    "                       (sign matching the B2 diagnostic tables)\n",
    "            roll_deg:  tilt about world X\n",
    "            up_z:      local-Z axis z component (1.0 = flat)\n",
    "        \"\"\"\n",
    "        mat = np.asarray(\n",
    "            self.data.site_xmat[self.left_foot_site]\n",
    "        ).reshape(3, 3)\n",
    "        z = mat[:, 2]\n",
    "        pitch_deg = float(math.degrees(math.atan2(-z[0], z[2])))\n",
    "        roll_deg = float(math.degrees(math.atan2(z[1], z[2])))\n",
    "        return pitch_deg, roll_deg, float(z[2])\n",
    "\n",
    "    def _ankle_trim_cmds(self, sw: float):\n",
    "        \"\"\"\n",
    "        V17-B4A fixed ankle-only orientation trim, phased by swing_env.\n",
    "\n",
    "        Returns (pitch_trim, roll_trim) in radians to add to\n",
    "        left_ankle_pitch / left_ankle_roll. Zero outside the fade\n",
    "        window, so it is inert for descent, touchdown and recovery.\n",
    "\n",
    "        Pure function of (config, sw): no hidden state, identical value\n",
    "        whether called from the controller or from _get_info.\n",
    "        \"\"\"\n",
    "        if (\n",
    "            not self.cfg.ankle_trim_enabled\n",
    "            or sw <= self.cfg.ankle_trim_fade_start\n",
    "        ):\n",
    "            return 0.0, 0.0\n",
    "\n",
    "        u = float(\n",
    "            np.clip(\n",
    "                (sw - self.cfg.ankle_trim_fade_start)\n",
    "                / max(\n",
    "                    self.cfg.ankle_trim_fade_end\n",
    "                    - self.cfg.ankle_trim_fade_start,\n",
    "                    1e-9,\n",
    "                ),\n",
    "                0.0,\n",
    "                1.0,\n",
    "            )\n",
    "        )\n",
    "        fade = self._smoothstep(u)\n",
    "\n",
    "        pitch_trim = -float(\n",
    "            np.clip(\n",
    "                self.cfg.ankle_trim_fraction\n",
    "                * self.cfg.ankle_trim_pitch_rad,\n",
    "                -self.cfg.ankle_trim_pitch_max,\n",
    "                self.cfg.ankle_trim_pitch_max,\n",
    "            )\n",
    "        ) * fade\n",
    "\n",
    "        roll_trim = -float(\n",
    "            np.clip(\n",
    "                self.cfg.ankle_trim_fraction\n",
    "                * self.cfg.ankle_trim_roll_rad,\n",
    "                -self.cfg.ankle_trim_roll_max,\n",
    "                self.cfg.ankle_trim_roll_max,\n",
    "            )\n",
    "        ) * fade\n",
    "\n",
    "        return float(pitch_trim), float(roll_trim)\n",
]

TRIM_BLOCK = [
    "\n",
    "        # --------------------------------------------------------------\n",
    "        # V17-B4A ANKLE-ONLY ORIENTATION TRIM (experimental)\n",
    "        # --------------------------------------------------------------\n",
    "        #\n",
    "        # The B2 position-only IK leaves the ankle joints free in the\n",
    "        # swing null-space, so the foot pitches ~9 deg and rolls ~4 deg\n",
    "        # near peak and a sole edge recontacts early at ~13 N, holding\n",
    "        # the site at ~23.1 mm instead of the commanded 26 mm.\n",
    "        #\n",
    "        # Full 6D pose IK (B3) re-routed hip/knee/ankle together and\n",
    "        # destroyed balance. B4A therefore touches ONLY the two ankle\n",
    "        # joints, with a small fixed trim derived from the measured\n",
    "        # flattening correction, gated above sw > 0.60.\n",
    "        #\n",
    "        # This is an additive overlay on the final target: the proven\n",
    "        # B2 support / sagittal / COM-hold laws are untouched.\n",
    "        _pitch_trim, _roll_trim = self._ankle_trim_cmds(sw)\n",
    "        if _pitch_trim != 0.0 or _roll_trim != 0.0:\n",
    "            target[4] += _pitch_trim     # left ankle pitch\n",
    "            target[5] += _roll_trim      # left ankle roll\n",
]

# Statements must be inserted BEFORE the ``return {`` of _get_info, never
# inside the dict literal. Dict entries go into a separate block.
INFO_PRE_BLOCK = [
    "\n",
    "        # V17-B4A ankle-trim + foot-orientation diagnostics.\n",
    "        _pitch_trim, _roll_trim = self._ankle_trim_cmds(sw)\n",
    "        _fp, _fr, _fupz = self._left_foot_orientation_diag()\n",
]

INFO_DICT_BLOCK = [
    "\n",
    "            \"left_foot_pitch_deg\": float(_fp),\n",
    "            \"left_foot_roll_deg\": float(_fr),\n",
    "            \"left_foot_up_z\": float(_fupz),\n",
    "            \"ankle_trim_pitch_deg\": float(\n",
    "                math.degrees(_pitch_trim)\n",
    "            ),\n",
    "            \"ankle_trim_roll_deg\": float(\n",
    "                math.degrees(_roll_trim)\n",
    "            ),\n",
    "            \"ankle_trim_active\": bool(\n",
    "                self.cfg.ankle_trim_enabled\n",
    "                and sw > self.cfg.ankle_trim_fade_start\n",
    "            ),\n",
]

EVAL_ARG_BLOCK = [
    "\n",
    "    # V17-B4A ankle-only orientation trim.\n",
    "    # 0.0 reproduces the trusted B2 behavior exactly (trim off).\n",
    "    parser.add_argument(\n",
    "        \"--ankle_trim_fraction\", type=float, default=0.0\n",
    "    )\n",
]

EVAL_KWARG_BLOCK = [
    "        ankle_trim_fraction=args.ankle_trim_fraction,\n",
]

EVAL_INIT_BLOCK = [
    "\n",
    "    # V17-B4A orientation diagnostics at the peak-clearance step.\n",
    "    peak_left_foot_pitch_deg = 0.0\n",
    "    peak_left_foot_roll_deg = 0.0\n",
    "    peak_left_foot_up_z = 1.0\n",
    "    max_ankle_trim_pitch_deg = 0.0\n",
    "    max_ankle_trim_roll_deg = 0.0\n",
]

EVAL_LOOP_BLOCK = [
    "\n",
    "        # V17-B4A: retain orientation at the first peak of clearance.\n",
    "        if float(info[\"left_foot_clearance\"]) > max_clear:\n",
    "            peak_left_foot_pitch_deg = float(\n",
    "                info.get(\"left_foot_pitch_deg\", 0.0)\n",
    "            )\n",
    "            peak_left_foot_roll_deg = float(\n",
    "                info.get(\"left_foot_roll_deg\", 0.0)\n",
    "            )\n",
    "            peak_left_foot_up_z = float(\n",
    "                info.get(\"left_foot_up_z\", 1.0)\n",
    "            )\n",
    "        max_ankle_trim_pitch_deg = max(\n",
    "            max_ankle_trim_pitch_deg,\n",
    "            abs(float(info.get(\"ankle_trim_pitch_deg\", 0.0))),\n",
    "        )\n",
    "        max_ankle_trim_roll_deg = max(\n",
    "            max_ankle_trim_roll_deg,\n",
    "            abs(float(info.get(\"ankle_trim_roll_deg\", 0.0))),\n",
    "        )\n",
]

EVAL_ROW_BLOCK = [
    '        "peak_left_foot_pitch_deg": peak_left_foot_pitch_deg,\n',
    '        "peak_left_foot_roll_deg": peak_left_foot_roll_deg,\n',
    '        "peak_left_foot_up_z": peak_left_foot_up_z,\n',
    '        "max_ankle_trim_pitch_deg": max_ankle_trim_pitch_deg,\n',
    '        "max_ankle_trim_roll_deg": max_ankle_trim_roll_deg,\n',
]

# The episode-summary print is ONE implicit-concatenated f-string:
#
#     print(
#         f"ep={ep:02d} "
#         ...
#         f"DIAG={row['diagnosis']}"
#     )
#
# so B4A values must be inserted as adjacent f-string lines BEFORE the
# DIAG line (no commas, each ending in a trailing space), never as a
# nested print(...).
EVAL_PRINT_BLOCK = [
    '            f"footPitch={row[\'peak_left_foot_pitch_deg\']:+.2f} "\n',
    '            f"footRoll={row[\'peak_left_foot_roll_deg\']:+.2f} "\n',
    '            f"footUpZ={row[\'peak_left_foot_up_z\']:.4f} "\n',
    '            f"trimP={row[\'max_ankle_trim_pitch_deg\']:.2f} "\n',
    '            f"trimR={row[\'max_ankle_trim_roll_deg\']:.2f} "\n',
]

EVAL_SUMMARY_BLOCK = [
    '        "peak_left_foot_pitch_deg",\n',
    '        "peak_left_foot_roll_deg",\n',
    '        "peak_left_foot_up_z",\n',
    '        "max_ankle_trim_pitch_deg",\n',
    '        "max_ankle_trim_roll_deg",\n',
]


def patch_env() -> None:
    text = ENV_PATH.read_text(encoding="utf-8")

    if ENV_MARKER in text:
        print(f"env   : already patched ({ENV_MARKER!r} present), no change")
        return

    if B3_MARKER in text:
        fail(f"working env is contaminated with B3 ({B3_MARKER!r}); aborting")

    check_signatures(text)

    if CHECKPOINT_PATH.exists():
        if ENV_PATH.read_bytes() == CHECKPOINT_PATH.read_bytes():
            print("env   : working file is byte-identical to trusted B2 checkpoint")
        else:
            print(
                "WARNING: working file differs from checkpoint, but B2 signatures "
                "are present and no B4A/B3 markers found -- continuing"
            )
    else:
        print(f"WARNING: checkpoint file missing: {CHECKPOINT_PATH.name}")

    lines = text.splitlines(keepends=True)

    lines, _ = insert_after(
        lines, "    swing_z_weight: float = 1.00", CONFIG_BLOCK
    )
    lines, _ = insert_before(
        lines, "    def _foot_contact(self, body_id: int) -> bool:",
        HELPERS_BLOCK,
    )
    lines, _ = insert_before(
        lines, "        # V13 SHARED-SUPPORT-LATCH RECOVERY", TRIM_BLOCK
    )
    lines, _ = insert_before(
        lines,
        "        return {",
        INFO_PRE_BLOCK,
    )
    lines, _ = insert_after(
        lines,
        '            "support_slip": float(right_support_displacement),',
        INFO_DICT_BLOCK,
    )

    ENV_PATH.write_text("".join(lines), encoding="utf-8")
    print("env   : patched -> V17-B4A ankle-only orientation trim installed")


def patch_eval() -> None:
    text = EVAL_PATH.read_text(encoding="utf-8")

    if EVAL_MARKER in text:
        print(f"eval  : already patched ({EVAL_MARKER!r} present), no change")
        return

    lines = text.splitlines(keepends=True)

    lines, _ = insert_after(
        lines,
        '    parser.add_argument("--swing_z_weight", type=float, default=1.00)',
        EVAL_ARG_BLOCK,
    )
    lines, _ = insert_after(
        lines, "        swing_z_weight=args.swing_z_weight,", EVAL_KWARG_BLOCK
    )
    lines, _ = insert_after(
        lines, "    max_support_slip = 0.0", EVAL_INIT_BLOCK
    )
    lines, _ = insert_before(
        lines,
        '        max_clear = max(max_clear, float(info["left_foot_clearance"]))',
        EVAL_LOOP_BLOCK,
    )
    lines, _ = insert_after(
        lines, '        "max_support_slip": max_support_slip,', EVAL_ROW_BLOCK
    )
    lines, _ = insert_after(
        lines, '            f"DIAG={row[\'diagnosis\']}"', EVAL_PRINT_BLOCK
    )
    lines, _ = insert_after(
        lines, '        "max_support_slip",', EVAL_SUMMARY_BLOCK
    )

    EVAL_PATH.write_text("".join(lines), encoding="utf-8")
    print("eval  : patched -> --ankle_trim_fraction + diagnostic columns")


def verify() -> None:
    env_text = ENV_PATH.read_text(encoding="utf-8")
    eval_text = EVAL_PATH.read_text(encoding="utf-8")

    checks = [
        ("env B4A trim block", ENV_MARKER in env_text),
        ("env helper _ankle_trim_cmds", "_ankle_trim_cmds" in env_text),
        ("env helper _left_foot_orientation_diag",
         "_left_foot_orientation_diag" in env_text),
        ("env config ankle_trim_fraction", "ankle_trim_fraction: float" in env_text),
        ("env NO B3", B3_MARKER not in env_text),
        ("eval trim arg", EVAL_MARKER in eval_text),
        ("eval peak pitch column", "peak_left_foot_pitch_deg" in eval_text),
    ]

    failed = [name for name, ok in checks if not ok]
    if failed:
        fail("post-patch verification failed: " + ", ".join(failed))

    for name, _ok in checks:
        print(f"verify: {name:<32s} OK")

    py_compile(ENV_PATH)
    py_compile(EVAL_PATH)
    print("verify: py_compile OK (env + evaluator)")


def main() -> int:
    print("=" * 78)
    print("V17-B4A STATE-AWARE PATCH (trusted B2 -> B4A ankle-only trim)")
    print("=" * 78)

    patch_env()
    patch_eval()
    verify()

    check_code = (
        "import sys; sys.path.insert(0, r'@ROOT@'); "
        "from envs.g1_deterministic_left_lift_env import "
        "G1DeterministicLeftLiftEnv, DeterministicLeftLiftConfig; "
        "cfg = DeterministicLeftLiftConfig(); "
        "print('import: G1DeterministicLeftLiftEnv OK'); "
        "print('import: fraction={:.2f} pitch_max_deg={:.2f} "
        "roll_max_deg={:.2f} fade={}/{}, enabled={}'.format("
        "cfg.ankle_trim_fraction, "
        "cfg.ankle_trim_pitch_max * 180.0 / 3.141592653589793, "
        "cfg.ankle_trim_roll_max * 180.0 / 3.141592653589793, "
        "cfg.ankle_trim_fade_start, cfg.ankle_trim_fade_end, "
        "cfg.ankle_trim_enabled))"
    ).replace("@ROOT@", str(ROOT))

    import_check = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
    )
    if import_check.returncode != 0:
        print(import_check.stdout)
        print(import_check.stderr)
        fail("import verification failed")

    print(import_check.stdout.strip())
    print("PATCH COMPLETE: one B4A episode is now ready to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
