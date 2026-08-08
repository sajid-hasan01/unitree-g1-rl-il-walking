from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from envs.g1_guarded_taskspace_right_lift_env import G1GuardedTaskspaceRightLiftEnv


def max_consecutive_true(flags):
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def rollout(args, c):
    env = G1GuardedTaskspaceRightLiftEnv(
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=c['cycle_duration'],
        settle_end=0.14,
        shift_end=c['shift_end'],
        guard_end=c['guard_end'],
        lift_end=c['lift_end'],
        land_end=c['land_end'],
        recover_end=1.00,
        target_clearance=c['target_clearance'],
        target_lateral_shift=c['target_lateral_shift'],
        ik_gain=c['ik_gain'],
        ik_damping=c['ik_damping'],
        ik_max_delta=c['ik_max_delta'],
        xy_hold_weight=c['xy_hold_weight'],
        z_lift_weight=c['z_lift_weight'],
        guard_x_abs=args.guard_x_abs,
        guard_y_abs=args.guard_y_abs,
        guard_x_velocity_abs=args.guard_x_velocity_abs,
        guard_y_velocity_abs=args.guard_y_velocity_abs,
        abort_x_velocity=c['abort_x_velocity'],
        abort_x_position=args.abort_x_position,
        abort_up_z=args.abort_up_z,
        x_hard_limit=args.x_hard_limit,
        y_hard_limit=args.y_hard_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        y_velocity_hard_limit=args.y_velocity_hard_limit,
        randomize_reset=False,
    )
    action = np.zeros(4, dtype=np.float32)
    obs, info = env.reset()
    steps = 0
    total = 0.0
    max_clear = 0.0
    min_up = 1.0
    contacts = []
    slips = []
    air_flags = []
    support_flags = []
    guard_seen = False
    abort_seen = False
    final = info
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        total += float(reward)
        max_clear = max(max_clear, float(info['main_clearance']))
        min_up = min(min_up, float(info['up_z']))
        contacts.append(float(info['contact_accuracy']))
        slips.append(float(info['support_slip']))
        guard_seen = guard_seen or bool(info['guard_passed_once'])
        abort_seen = abort_seen or bool(info['abort_lift'])
        if float(info['main_target_clearance']) >= args.strict_clearance:
            air_flags.append(not bool(info['right_contact']))
            support_flags.append(bool(info['left_contact']))
        final = info

    reason = env.termination_reason(final) if steps < args.max_steps else 'max_steps'
    air_steps = int(sum(air_flags))
    max_air_streak = int(max_consecutive_true(air_flags))
    support_ok_ratio = float(np.mean(support_flags)) if support_flags else 0.0
    landing_ok = int(bool(final['left_contact']) and bool(final['right_contact']))
    strict = int(
        steps >= args.max_steps
        and reason == 'max_steps'
        and max_clear >= args.strict_clearance
        and air_steps >= args.min_air_steps
        and max_air_streak >= args.min_air_streak
        and support_ok_ratio >= args.min_support_ratio
        and min_up >= args.strict_min_up
        and abs(float(final['x_position'])) <= args.strict_x_abs
        and abs(float(final['y_position'])) <= args.strict_y_abs
        and landing_ok == 1
        and guard_seen
    )
    score = (
        1600.0 * strict
        + steps
        + 13000.0 * min(max_clear, 0.035)
        + 12.0 * air_steps
        + 10.0 * max_air_streak
        + 150.0 * min_up
        + 120.0 * support_ok_ratio
        + 180.0 * landing_ok
        + 100.0 * int(guard_seen)
        - 480.0 * abs(float(final['x_position']))
        - 450.0 * abs(float(final['y_position']))
        - 180.0 * abs(float(final['x_velocity']))
        - 180.0 * abs(float(final['y_velocity']))
        - 300.0 * float(np.mean(slips) if slips else 0.0)
        - 120.0 * int(abort_seen)
    )
    row = dict(c)
    row.update(
        steps=steps,
        reward=total,
        main_clearance=max_clear,
        min_up_z=min_up,
        air_steps=air_steps,
        max_air_streak=max_air_streak,
        support_ok_ratio=support_ok_ratio,
        landing_ok=landing_ok,
        guard_seen=int(guard_seen),
        abort_seen=int(abort_seen),
        final_state=str(final['state_name']),
        contact=float(np.mean(contacts)) if contacts else 0.0,
        support_slip=float(np.mean(slips)) if slips else 0.0,
        final_x=float(final['x_position']),
        final_y=float(final['y_position']),
        final_x_velocity=float(final['x_velocity']),
        final_y_velocity=float(final['y_velocity']),
        left_contact=int(bool(final['left_contact'])),
        right_contact=int(bool(final['right_contact'])),
        strict_pass=strict,
        score=score,
        reason=reason,
    )
    env.close()
    return row


def print_row(r):
    print(
        f"{r['name']:<10s} C={r['target_clearance']:.3f} Y={r['target_lateral_shift']:.3f} "
        f"dur={r['cycle_duration']:.1f} shift={r['shift_end']:.2f} guard={r['guard_end']:.2f} "
        f"lift={r['lift_end']:.2f} land={r['land_end']:.2f} ik={r['ik_gain']:.2f} "
        f"abortV={r['abort_x_velocity']:+.2f} | steps={r['steps']:04d} "
        f"clear={r['main_clearance']:.4f} air={r['air_steps']:03d}/{r['max_air_streak']:03d} "
        f"up={r['min_up_z']:.3f} x={r['final_x']:+.3f} y={r['final_y']:+.3f} "
        f"xv={r['final_x_velocity']:+.3f} yv={r['final_y_velocity']:+.3f} "
        f"landOK={r['landing_ok']} guard={r['guard_seen']} abort={r['abort_seen']} "
        f"L={r['left_contact']} R={r['right_contact']} strict={r['strict_pass']} "
        f"score={r['score']:+.1f} reason={r['reason']}"
    )


def main():
    parser = argparse.ArgumentParser(description='Fast sweep for guarded finite-state task-space right-foot lift.')
    parser.add_argument('--model_path', type=str, default='third_party/mujoco_menagerie/unitree_g1/scene.xml')
    parser.add_argument('--csv', type=str, default='results/guarded_taskspace_right_lift_sweep.csv')
    parser.add_argument('--frame_skip', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=520)
    parser.add_argument('--guard_x_abs', type=float, default=0.080)
    parser.add_argument('--guard_y_abs', type=float, default=0.060)
    parser.add_argument('--guard_x_velocity_abs', type=float, default=0.280)
    parser.add_argument('--guard_y_velocity_abs', type=float, default=0.350)
    parser.add_argument('--abort_x_position', type=float, default=-0.18)
    parser.add_argument('--abort_up_z', type=float, default=0.88)
    parser.add_argument('--x_hard_limit', type=float, default=0.36)
    parser.add_argument('--y_hard_limit', type=float, default=0.32)
    parser.add_argument('--x_velocity_hard_limit', type=float, default=1.35)
    parser.add_argument('--y_velocity_hard_limit', type=float, default=1.35)
    parser.add_argument('--strict_clearance', type=float, default=0.020)
    parser.add_argument('--min_air_steps', type=int, default=5)
    parser.add_argument('--min_air_streak', type=int, default=3)
    parser.add_argument('--min_support_ratio', type=float, default=0.90)
    parser.add_argument('--strict_min_up', type=float, default=0.86)
    parser.add_argument('--strict_x_abs', type=float, default=0.22)
    parser.add_argument('--strict_y_abs', type=float, default=0.20)
    args = parser.parse_args()

    candidates = []
    i = 0
    for clearance, lift_end, abort_v, ik in product([0.025, 0.030, 0.035], [0.56, 0.58, 0.60], [-0.45, -0.55], [1.00, 1.15]):
        candidates.append({
            'name': f'G{i:02d}',
            'cycle_duration': 5.2,
            'shift_end': 0.42,
            'guard_end': 0.50,
            'lift_end': lift_end,
            'land_end': 0.72,
            'target_clearance': clearance,
            'target_lateral_shift': 0.018,
            'ik_gain': ik,
            'ik_damping': 0.045,
            'ik_max_delta': 0.20,
            'xy_hold_weight': 0.05,
            'z_lift_weight': 1.20,
            'abort_x_velocity': abort_v,
        })
        i += 1
    for clearance, guard_end, lift_end, land_end in [(0.025,0.48,0.54,0.66),(0.030,0.48,0.54,0.66),(0.030,0.50,0.56,0.68),(0.035,0.50,0.56,0.68)]:
        candidates.append({
            'name': f'G{i:02d}',
            'cycle_duration': 5.2,
            'shift_end': 0.42,
            'guard_end': guard_end,
            'lift_end': lift_end,
            'land_end': land_end,
            'target_clearance': clearance,
            'target_lateral_shift': 0.018,
            'ik_gain': 1.20,
            'ik_damping': 0.045,
            'ik_max_delta': 0.22,
            'xy_hold_weight': 0.05,
            'z_lift_weight': 1.25,
            'abort_x_velocity': -0.45,
        })
        i += 1

    print('=' * 132)
    print('GUARDED TASK-SPACE RIGHT-LIFT SWEEP')
    print(f'Candidates: {len(candidates)}')
    print('Goal: short visible pop-up with landing before backward collapse.')
    print('=' * 132)

    rows = []
    for c in candidates:
        r = rollout(args, c)
        rows.append(r)
        print_row(r)

    rows_sorted = sorted(rows, key=lambda r: float(r['score']), reverse=True)
    print('\nTOP 10 BY SCORE')
    for r in rows_sorted[:10]:
        print_row(r)
    passing = [r for r in rows_sorted if int(r['strict_pass']) == 1]
    print('\nSTRICT PASSING ROWS')
    if passing:
        for r in passing:
            print_row(r)
    else:
        print('None')
    air_rows = [r for r in rows_sorted if int(r['air_steps']) > 0]
    print('\nBEST ROWS WITH AIR-TIME')
    if air_rows:
        for r in air_rows[:10]:
            print_row(r)
    else:
        print('None')

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('\nCSV saved:', args.csv)


if __name__ == '__main__':
    main()
