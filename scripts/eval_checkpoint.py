"""Measure the lift-success rate of a trained rsl_rl checkpoint.

Use this as a checkpoint/runtime compatibility preflight for the Lift-Cube PPO
policy. PPO checkpoints can vary substantially by training stage and simulator
version, so measure success directly instead of treating the checkpoint as
competent by default or trusting the reward curve.

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
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval rsl_rl checkpoint lift success")
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--lift_margin", type=float, default=0.06)
parser.add_argument(
    "--sustain_steps",
    type=int,
    default=10,
    help="Required consecutive lifted steps at the end of evaluation.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--asset_root", default=None)
parser.add_argument("--json", dest="json_path", default=None, help="Write machine-readable result or failure JSON")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1:
    parser.error("--num_envs must be >= 1")
if args.sustain_steps < 1:
    parser.error("--sustain_steps must be >= 1")
if args.steps < args.sustain_steps:
    parser.error("--steps must be >= --sustain_steps")
if args.lift_margin <= 0:
    parser.error("--lift_margin must be > 0")
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import numpy as np
import torch
import warp as wp
import gymnasium as gym

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from ipfd.adapters.isaac_lab import configure_asset_root
from ipfd import __version__
from ipfd.oracles.rsl_rl_policy import checkpoint_sha256, load_learned_policy
from ipfd.provenance import source_provenance

configure_asset_root(args.asset_root)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg


def obj_z_all(env) -> torch.Tensor:
    pos = wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)
    return pos[:, 2] - env.unwrapped.scene.env_origins[:, 2]


def main() -> None:
    import json
    env = None
    result = {
        "schema": "ipfd.competence.v1",
        "status": "failed",
        "task": args.task,
        "checkpoint": os.path.basename(args.checkpoint),
        "seed": args.seed,
        "n_episodes": args.num_envs,
        "steps": args.steps,
        "lift_margin_m": args.lift_margin,
        "sustain_steps": args.sustain_steps,
        "success_definition": "sustained_final_lift_v1",
    }

    def write_result() -> None:
        if not args.json_path:
            return
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    try:
        result["checkpoint_sha256"] = checkpoint_sha256(args.checkpoint)
        result["software"] = {
            "ipfd_version": __version__,
            **source_provenance(_REPO),
        }
        result["runtime"] = {
            package: metadata.version(package)
            for package in (
                "isaaclab",
                "isaaclab_tasks",
                "isaaclab_rl",
                "isaacsim",
                "torch",
                "rsl-rl-lib",
                "gymnasium",
                "warp-lang",
            )
        }
        torch.manual_seed(args.seed)
        env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)
        device = str(env.unwrapped.device)
        policy = load_learned_policy(env, agent_cfg.to_dict(), args.checkpoint, device=device)

        settle = 30
        env.reset()
        zs = []
        zeros = torch.zeros(
            (env.num_envs, env.action_space.shape[1]),
            device=device,
        )
        for _ in range(settle):
            env.step(zeros)
            zs.append(obj_z_all(env).clone())
        z_rest = torch.stack(zs).min(dim=0).values

        env.reset()
        for _ in range(15):
            env.step(zeros)
        obs = env.get_observations()
        max_z = torch.full((env.num_envs,), -10.0, device=device)
        final_streak = torch.zeros((env.num_envs,), dtype=torch.int64, device=device)
        for _step in range(args.steps):
            actions = policy(obs)
            obs, _r, _dones, _e = env.step(actions)
            policy.reset(_dones)
            z = obj_z_all(env)
            max_z = torch.maximum(max_z, z)
            lifted = z - z_rest > args.lift_margin
            final_streak = torch.where(lifted, final_streak + 1, 0)
        ml = (max_z - z_rest).detach().cpu().numpy()
        print("\n=== EVAL ===")
        print(f"checkpoint: {os.path.basename(args.checkpoint)}")
        print(f"max_lift  mean={ml.mean():.3f}  median={np.median(ml):.3f}  p90={np.percentile(ml,90):.3f}  max={ml.max():.3f}")
        for thr in (0.02, 0.04, 0.06, 0.10):
            print(f"  frac lifted >{thr:.2f}m at some point: {(ml > thr).mean():.2%}")
        successful = final_streak >= args.sustain_steps
        success_rate = float(successful.float().mean().item())
        print(
            f"SUCCESS_RATE (>{args.lift_margin:.2f}m for final "
            f"{args.sustain_steps} steps): {success_rate:.2%}"
        )
        result.update({
            "status": "complete",
            "success_rate": success_rate,
            "mean_lift": float(ml.mean()),
            "max_lift": float(ml.max()),
        })
        # Write before the finally block: simulation_app.close() can hard-exit
        # the process, so anything after it is not guaranteed to run.
        write_result()
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
        print(f"EVAL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        write_result()
        raise SystemExit(2) from None
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
