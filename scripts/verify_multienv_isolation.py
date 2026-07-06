"""Pivotal test: does reset_to(env_ids=[k]) poison OTHER envs?

verify_pnor_decoupled.py confirmed a single reset_to() permanently corrupts the
sim and the corruption survives env.reset() (reset_to_poisons_env: YES). Isaac
Lab runs ONE global SimulationContext per process, so a truly separate sim
instance per probe is not available in-process. The only light isolation is a
VECTORISED env: keep the primary rollout in env 0 and farm recovery probes to
envs 1..N-1, resetting ONLY their env_ids. That works iff per-env reset_to is
LOCAL -- i.e. resetting env 1 does not disturb env 0.

This script measures exactly that, with num_envs=2:

  CONTROL phase: drive both envs with the primary pick-lift, NO reset_to. Record
    env 0's max cube lift (baseline: should match the ~0.3 m single-env diagnose).

  PROBED phase: reset, save an early state S, drive both envs, and every K steps
    reset_to(S, env_ids=[1]) -- churning env 1 through contact-rich restores.
    Record env 0's max lift. If env 0 still lifts ~like CONTROL, per-env reset_to
    is local and probe isolation via a multi-env pool is viable. If env 0's lift
    collapses, reset_to poisons globally and probe-based PoNR is not achievable in
    this Isaac Lab version without a subprocess per probe.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_multienv_isolation.py --headless
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD multi-env reset_to isolation test")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=150)
parser.add_argument("--reset_stride", type=int, default=10, help="Reset env 1 every N steps in PROBED phase.")
parser.add_argument("--lift_thresh", type=float, default=0.04)
parser.add_argument("--tol", type=float, default=0.05, help="Max acceptable env0 lift drop vs control [m].")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))
wp.init()
from _lift_sm import PickAndLiftSm, sm_action  # noqa: E402
from ipfd.adapters.isaac_lab import slice_state  # noqa: E402


def log(msg: str) -> None:
    print(f"[isolation] {msg}", flush=True)


def n_identity(n: int, dev) -> torch.Tensor:
    a = torch.zeros((n, 8), dtype=torch.float32, device=dev)
    a[:, 3] = 1.0
    return a


def obj_z_all(env) -> torch.Tensor:
    return wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)[:, 2].clone()


def rest_height(objz_hist: list[torch.Tensor], n: int) -> torch.Tensor:
    """Per-env settled rest height = min over the first 15 recorded steps."""
    stack = torch.stack(objz_hist[: min(15, len(objz_hist))], dim=0)  # (steps, n)
    return stack.min(dim=0).values


def run_phase(env, dt, dev, n, seed, probe_env1: bool):
    des_or = torch.zeros((n, 4), device=dev)
    des_or[:, 1] = 1.0
    env.reset(seed=seed)
    env.step(n_identity(n, dev))
    sm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)

    S = None
    reset_ids = torch.tensor([1], device=dev, dtype=torch.long)
    objz_hist = [obj_z_all(env)]
    zmax = obj_z_all(env)
    action = n_identity(n, dev)
    n_resets = 0
    for step in range(args.max_steps):
        if probe_env1 and step == 5:
            # early (pre-grasp) reference for env 1, sliced to env_ids for reset_to
            S = slice_state(env.unwrapped.scene.get_state(), slice(1, 2))
        env.step(action)
        z = obj_z_all(env)
        zmax = torch.maximum(zmax, z)
        objz_hist.append(z)
        if probe_env1 and S is not None and step > 5 and step % args.reset_stride == 0:
            env.unwrapped.scene.reset_to(S, reset_ids)
            n_resets += 1
        action = sm_action(env, sm, des_or)
    rest = rest_height(objz_hist, n)
    lift = (zmax - rest).detach().cpu().numpy()
    return lift, n_resets


def main() -> None:
    env = None
    result = {"per_env_reset_to_is_local": "UNKNOWN", "isolation_viable": "UNKNOWN"}
    try:
        n = max(2, args.num_envs)
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=n)
        env = gym.make(args.env_id, cfg=env_cfg)
        dev = env.unwrapped.device
        dt = env_cfg.sim.dt * env_cfg.decimation
        log(f"env={args.env_id} num_envs={n} dt={dt:.4f}")

        # CONTROL first (no reset_to anywhere -> no global poison carried in).
        lift_ctrl, _ = run_phase(env, dt, dev, n, seed=0, probe_env1=False)
        log(f"CONTROL   env-lift [m] = {np.array2string(lift_ctrl, precision=3)} "
            f"(env0={lift_ctrl[0]:+.3f})")

        # PROBED: churn env 1 through reset_to; measure env 0.
        lift_probe, n_resets = run_phase(env, dt, dev, n, seed=0, probe_env1=True)
        log(f"PROBED    env-lift [m] = {np.array2string(lift_probe, precision=3)} "
            f"(env0={lift_probe[0]:+.3f}, env1_resets={n_resets})")

        env0_drop = float(lift_ctrl[0] - lift_probe[0])
        env0_still_lifts = lift_probe[0] > args.lift_thresh
        local = (env0_drop <= args.tol) and env0_still_lifts
        log(f"env0 lift drop (control - probed) = {env0_drop:+.3f} m  (tol {args.tol}); "
            f"env0 still lifts = {env0_still_lifts}")
        result["per_env_reset_to_is_local"] = "YES" if local else "NO"
        result["isolation_viable"] = "YES" if local else "NO"
        if local:
            log("=> per-env reset_to is LOCAL: primary in env 0 + probe pool in envs 1..N is viable.")
        else:
            log("=> reset_to disturbs env 0: multi-env isolation does NOT work; "
                "probe-based PoNR needs a subprocess per probe or an upstream fix.")
    except Exception:
        import traceback
        log("aborted with exception:")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        print("\n" + "=" * 55)
        print("MULTIENV_ISOLATION:")
        for k in ("per_env_reset_to_is_local", "isolation_viable"):
            print(f"- {k}: {result[k]}")
        print("=" * 55, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
