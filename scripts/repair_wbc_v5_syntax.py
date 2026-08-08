from pathlib import Path

p = Path("envs/g1_wbc_taskspace_right_lift_env.py")
s = p.read_text(encoding="utf-8")

bad = "self._touchdown_timer self._touchdown_force"
if bad not in s:
    print("The known corrupted pattern was not found. Checking compile next.")
else:
    lines = s.splitlines()
    fixed = []
    for line in lines:
        if bad in line:
            indent = line[:len(line) - len(line.lstrip())]
            fixed.append(indent + "self._touchdown_force = float(np.clip(1.0 - self._touchdown_timer / 0.30, 0.0, 1.0))")
        else:
            fixed.append(line)
    p.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    print("Fixed corrupted touchdown_force line.")

