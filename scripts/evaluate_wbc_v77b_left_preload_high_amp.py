from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.evaluate_wbc_v77a_left_preload_diag import run_case


def main() -> None:
    out_csv = Path(
        "experiments/amass_b3_15dof_residual_ppo_v3b/reports/wbc_v77b_left_preload_high_amp.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("right_support_c", +1.0),
        ("right_support_d", -1.0),
    ]

    amps = [0.080, 0.100, 0.120, 0.140, 0.160, 0.180, 0.200, 0.240]

    rows = []

    print("=" * 160)
    print("WBC V7.7B HIGH-AMPLITUDE LEFT PRELOAD SWEEP")
    print("Goal: push the best left-preload template harder until Lratio drops enough.")
    print("=" * 160)

    for template, sign in configs:
        for amp in amps:
            row = run_case(template, sign, amp)
            rows.append(row)

            print(
                f"tpl={row['template']:<16} "
                f"sign={row['preload_sign']:+.0f} "
                f"amp={row['preload_amp']:.3f} "
                f"steps={row['steps']:04d} "
                f"reason={row['reason']:<16} "
                f"Lforce={row['left_force_min']:.2f} "
                f"Rforce={row['right_force_at_min']:.2f} "
                f"Lratio={row['left_force_ratio']:.3f} "
                f"Lclear={row['left_clearance']:.4f} "
                f"Rclear={row['right_clearance']:.4f} "
                f"up={row['min_up_z']:.3f} "
                f"slip={row['support_slip']:.4f} "
                f"ang={row['max_root_ang_vel']:.3f} "
                f"x={row['final_x']:+.3f} "
                f"y={row['final_y']:+.3f} "
                f"xv={row['final_x_velocity']:+.3f}"
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("CSV saved:", out_csv)


if __name__ == "__main__":
    main()
