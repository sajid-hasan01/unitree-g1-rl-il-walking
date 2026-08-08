from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mujoco.viewer
import numpy as np

from envs.g1_guarded_taskspace_right_lift_env import G1GuardedTaskspaceRightLiftEnv


def parse_action(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(',') if x.strip()]
    if len(vals) != 4:
        raise ValueError('Action must contain exactly 4 comma-separated values.')
    return np.asarray(vals, dtype=np.float32)


def add_args(parser):
    parser.add_argument('--model_path', type=str, default='third_party/mujoco_menagerie/unitree_g1/scene.xml')
    parser.add_argument('--fixed_action', type=str, default='0,0,0,0')
    parser.add_argument('--print_every', type=int, default=25)
    parser.add_argument('--sleep_time', type=float, default=0.01)
    parser.add_argument('--frame_skip', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=520)
    parser.add_argument('--cycle_duration', type=float, default=5.2)
    parser.add_argument('--settle_end', type=float, default=0.14)
    parser.add_argument('--shift_end', type=float, default=0.42)
    parser.add_argument('--guard_end', type=float, default=0.50)
    parser.add_argument('--lift_end', type=float, default=0.60)
    parser.add_argument('--land_end', type=float, default=0.74)
    parser.add_argument('--recover_end', type=float, default=1.00)
    parser.add_argument('--target_clearance', type=float, default=0.030)
    parser.add_argument('--target_lateral_shift', type=float, default=0.018)
    parser.add_argument('--ik_gain', type=float, default=1.10)
    parser.add_argument('--ik_damping', type=float, default=0.045)
    parser.add_argument('--ik_max_delta', type=float, default=0.20)
    parser.add_argument('--xy_hold_weight', type=float, default=0.05)
    parser.add_argument('--z_lift_weight', type=float, default=1.20)
    parser.add_argument('--guard_x_abs', type=float, default=0.080)
    parser.add_argument('--guard_y_abs', type=float, default=0.060)
    parser.add_argument('--guard_x_velocity_abs', type=float, default=0.280)
    parser.add_argument('--guard_y_velocity_abs', type=float, default=0.350)
    parser.add_argument('--abort_x_velocity', type=float, default=-0.55)
    parser.add_argument('--abort_x_position', type=float, default=-0.18)
    parser.add_argument('--abort_up_z', type=float, default=0.88)
    parser.add_argument('--x_hard_limit', type=float, default=0.36)
    parser.add_argument('--y_hard_limit', type=float, default=0.32)
    parser.add_argument('--x_velocity_hard_limit', type=float, default=1.35)
    parser.add_argument('--y_velocity_hard_limit', type=float, default=1.35)


def make_env(args):
    return G1GuardedTaskspaceRightLiftEnv(
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        cycle_duration=args.cycle_duration,
        settle_end=args.settle_end,
        shift_end=args.shift_end,
        guard_end=args.guard_end,
        lift_end=args.lift_end,
        land_end=args.land_end,
        recover_end=args.recover_end,
        target_clearance=args.target_clearance,
        target_lateral_shift=args.target_lateral_shift,
        ik_gain=args.ik_gain,
        ik_damping=args.ik_damping,
        ik_max_delta=args.ik_max_delta,
        xy_hold_weight=args.xy_hold_weight,
        z_lift_weight=args.z_lift_weight,
        guard_x_abs=args.guard_x_abs,
        guard_y_abs=args.guard_y_abs,
        guard_x_velocity_abs=args.guard_x_velocity_abs,
        guard_y_velocity_abs=args.guard_y_velocity_abs,
        abort_x_velocity=args.abort_x_velocity,
        abort_x_position=args.abort_x_position,
        abort_up_z=args.abort_up_z,
        x_hard_limit=args.x_hard_limit,
        y_hard_limit=args.y_hard_limit,
        x_velocity_hard_limit=args.x_velocity_hard_limit,
        y_velocity_hard_limit=args.y_velocity_hard_limit,
        randomize_reset=False,
    )


def main():
    parser = argparse.ArgumentParser(description='Demo guarded task-space right-foot lift.')
    add_args(parser)
    args = parser.parse_args()
    action = parse_action(args.fixed_action)
    env = make_env(args)

    print('=' * 118)
    print('GUARDED TASK-SPACE RIGHT-FOOT LIFT DEMO')
    print('Fixed action:', action.tolist())
    print('Viewer opening...')
    print('=' * 118)

    obs, info = env.reset()
    total = 0.0
    max_clear = 0.0
    min_up = 1.0
    air_steps = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            if not viewer.is_running():
                break
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            total += float(reward)
            max_clear = max(max_clear, float(info['main_clearance']))
            min_up = min(min_up, float(info['up_z']))
            if float(info['main_target_clearance']) >= 0.020 and not bool(info['right_contact']):
                air_steps += 1
            if step % args.print_every == 0:
                print(
                    f"step={step:04d} state={info['state_name']:<7s} phi={info['phase']:.2f} "
                    f"guard={int(info['guard_passed_once'])}/{int(info['guard_ok_now'])} "
                    f"abort={int(info['abort_lift'])} swing={info['swing_env']:.2f} "
                    f"shift={info['shift_env']:.2f} x={info['x_position']:+.3f} "
                    f"y={info['y_position']:+.3f}/{info['target_y_offset']:+.3f} "
                    f"xv={info['x_velocity']:+.3f} yv={info['y_velocity']:+.3f} "
                    f"h={info['base_height']:.3f} up={info['up_z']:.3f} "
                    f"Rclr={info['right_foot_clearance']:.4f}/{info['main_target_clearance']:.4f} "
                    f"L={int(info['left_contact'])}/exp={int(info['left_expected_contact'])} "
                    f"R={int(info['right_contact'])}/exp={int(info['right_expected_contact'])} "
                    f"contact={info['contact_accuracy']:.2f} slip={info['support_slip']:.3f} "
                    f"rew={reward:+.3f}"
                )
            if terminated or truncated:
                print('\nEpisode ended')
                print('terminated:', terminated)
                print('truncated:', truncated)
                print('reason:', env.termination_reason(info) if terminated else 'max_steps')
                break
            if args.sleep_time > 0:
                time.sleep(args.sleep_time)

    print('\nRESULT')
    print('steps:', int(info['episode_step']))
    print('total_reward:', total)
    print('max_right_clearance:', max_clear)
    print('min_up_z:', min_up)
    print('air_steps_when_target_clearance_ge_2cm:', air_steps)
    print('final_info:', info)
    env.close()


if __name__ == '__main__':
    main()
