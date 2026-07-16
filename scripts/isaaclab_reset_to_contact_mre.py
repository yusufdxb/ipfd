"""Minimal reproducible example: scene.reset_to(scene.get_state()) is not a no-op
after contact -- the restored trajectory diverges from the un-restored one, even
though get_state() round-trips joint/object pose and velocity bit-exactly.

This is a STANDALONE Isaac Lab repro (no third-party package imports) suitable for
attaching to an Isaac Lab GitHub issue. It isolates the divergence to sim state
that scene.get_state() does not expose (the PhysX contact-manifold / solver
warm-start cache).

--------------------------------------------------------------------------------
EXPECTED behaviour
    Let S = scene.get_state() at some step. Restoring that exact state with
    scene.reset_to(S) should leave the simulation unchanged, so continuing with an
    identical (deterministic) action stream must produce an IDENTICAL trajectory,
    whether or not reset_to(S) was called. Formally, reset_to(get_state()) is the
    identity on the simulation.

OBSERVED behaviour
    Before contact, reset_to(get_state()) is (near) transparent. AFTER the gripper
    has grasped the cube, the two continuations diverge immediately and grow apart
    by >1e-1 in the policy observation within a few steps -- despite get_state()
    reporting the object/joint state was restored to < 1e-6. The unrestored piece
    of state is the contact/solver cache, which is not part of scene.get_state().

    This makes any save/restore-based analysis (e.g. a recovery probe that rolls
    the sim back to a checkpoint) silently corrupt once contact is involved, unless
    the probe is run in a SEPARATE environment instance.

EXACT ENVIRONMENT (fill in yours when filing)
    Isaac Lab 4.5.22 ; Isaac Sim 4.5 ; task Isaac-Lift-Cube-Franka-v0
    PhysX GPU pipeline ; num_envs=1 ; single CUDA GPU ; Ubuntu 22.04 ; Python 3.10

RUN
    OMNI_KIT_ACCEPT_EULA=YES <isaac-python> scripts/isaaclab_reset_to_contact_mre.py --headless
"""

from __future__ import annotations

# ruff: noqa: E402, I001  -- AppLauncher must run before any isaaclab import.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--grasp_steps", type=int, default=40, help="Policy steps to reach a grasp.")
parser.add_argument("--compare_steps", type=int, default=12, help="Steps compared across the reset.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def obs_vec(o) -> np.ndarray:
    # get_observations() and step() both return a TensorDict with a "policy" group
    # (a leaf tensor of shape (num_envs, obs_dim)); pull env 0's row as float64.
    return o["policy"][0].detach().cpu().numpy().reshape(-1).astype(np.float64)



def state_maxabs_diff(a, b) -> float:
    """Max abs difference over every leaf tensor of two nested sim states."""
    if hasattr(a, "shape") and hasattr(a, "detach"):
        return float((a.detach() - b.detach()).abs().max().item()) if a.numel() else 0.0
    if isinstance(a, dict):
        return max((state_maxabs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        return max((state_maxabs_diff(x, y) for x, y in zip(a, b, strict=False)), default=0.0)
    return 0.0


def main() -> None:
    torch.manual_seed(args.seed)
    env = gym.make(args.task, cfg=parse_env_cfg(args.task, num_envs=1))
    env = RslRlVecEnvWrapper(env)
    dev = env.unwrapped.device

    # Deterministic policy: NVIDIA's published checkpoint, mean (inference) action.
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from rsl_rl.runners import OnPolicyRunner
    import importlib.metadata as md
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    ckpt = get_published_pretrained_checkpoint("rsl_rl", args.task)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), md.version("rsl-rl-lib")
    ).to_dict()
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=str(dev))
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=str(dev))

    def act(o):
        with torch.inference_mode():
            return policy(o).clone()

    scene = env.unwrapped.scene
    obs = env.get_observations()
    for _ in range(args.grasp_steps):  # drive into a grasp
        obs, *_ = env.step(act(obs))

    S = _clone(scene.get_state())  # snapshot the grasped state

    # (A) Natural continuation from the grasped state -- no reset_to.
    obs_a, traj_a = obs, []
    for _ in range(args.compare_steps):
        obs_a, *_ = env.step(act(obs_a))
        traj_a.append(obs_vec(obs_a))

    # Restore the EXACT state we snapshotted, then verify get_state() round-trips it.
    scene.reset_to(S, torch.tensor([0], device=dev))
    round_trip = state_maxabs_diff(S, _clone(scene.get_state()))

    # (B) Continuation after reset_to(S) -- identical actions, identical state.
    obs_b, traj_b = env.get_observations(), []
    for _ in range(args.compare_steps):
        obs_b, *_ = env.step(act(obs_b))
        traj_b.append(obs_vec(obs_b))

    gaps = [float(np.abs(a - b).max()) for a, b in zip(traj_a, traj_b, strict=False)]

    print("\n================ Isaac Lab reset_to contact-state MRE ================")
    print(f"task                         : {args.task}  (num_envs=1)")
    print(f"visible state round-trip     : max|get_state()-S| = {round_trip:.2e}  (expected ~0)")
    print(f"per-step max obs gap A vs B  : {[f'{g:.2e}' for g in gaps]}")
    print(f"final obs gap (step {args.compare_steps:2d})       : {gaps[-1]:.3e}")
    reproduced = round_trip < 1e-4 and gaps[-1] > 1e-2
    print("\nEXPECTED: gaps are ~0 (reset_to(get_state()) is the identity).")
    print("OBSERVED: gaps grow despite an exact state round-trip -> contact/solver")
    print("          state is not captured by scene.get_state().")
    print(f"\nRESULT: {'BUG REPRODUCED' if reproduced else 'not reproduced (adjust grasp_steps?)'}")

    env.close()
    app.close()


def _clone(x):
    if hasattr(x, "clone"):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    return x


if __name__ == "__main__":
    main()
