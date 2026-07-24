"""Measure the lift-success rate of a trained rsl_rl checkpoint.

Used to pick a genuinely competent policy for the Phase 3 IPFD run: Lift-Cube PPO
is unstable and a late checkpoint can be worse than a mid-training peak, so we
measure success directly instead of trusting the reward curve.

Success = the cube is held above (settled rest height + margin) at the end of the
episode horizon, measured per parallel env.

Run:
    OMNI_KIT_ACCEPT_EULA=YES /path/to/isaac-lab/python \\
        scripts/eval_checkpoint.py --headless --checkpoint <model_*.pt>
"""

from __future__ import annotations

# ruff: noqa: I001, E402

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval rsl_rl checkpoint lift success")
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lift_margin", type=float, default=0.06)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import numpy as np
import torch
import warp as wp
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from ipfd.oracles.rsl_rl_policy import load_learned_policy


def obj_z_all(env) -> torch.Tensor:
    pos = wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)
    return pos[:, 2] - env.unwrapped.scene.env_origins[:, 2]


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    device = str(env.unwrapped.device)
    policy = load_learned_policy(env, agent_cfg.to_dict(), args.checkpoint, device=device)

    settle = 30  # steps to let the spawned cube fall and settle before measuring
    env.reset()
    obs = env.get_observations()
    zs = []  # per-step object z during settle window -> rest height
    max_z = torch.full((env.num_envs,), -10.0, device=device)
    for step in range(args.steps):
        actions = policy(obs)
        obs, _r, dones, _e = env.step(actions)
        z = obj_z_all(env)
        if step < settle:
            zs.append(z.clone())
        else:
            max_z = torch.maximum(max_z, z)  # peak AFTER settling

    z_rest = torch.stack(zs).min(dim=0).values  # settled table height per env
    ml = (max_z - z_rest).detach().cpu().numpy()
    print("\n=== EVAL ===")
    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"max_lift  mean={ml.mean():.3f}  median={np.median(ml):.3f}  p90={np.percentile(ml,90):.3f}  max={ml.max():.3f}")
    for thr in (0.02, 0.04, 0.06, 0.10):
        print(f"  frac lifted >{thr:.2f}m at some point: {(ml > thr).mean():.2%}")
    print(f"SUCCESS_RATE (>{args.lift_margin:.2f}m): {(ml > args.lift_margin).mean():.2%}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
