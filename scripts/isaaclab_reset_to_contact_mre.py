"""Minimal reproducer for a scene save/restore replay-equivalence question.

The restored trajectory diverges from the un-restored trajectory even though
exposed scene state round-trips bit-exactly and both branches receive the exact
same recorded action tensors.

This is a STANDALONE Isaac Lab repro (no third-party package imports) suitable for
attaching to an Isaac Lab GitHub issue. It demonstrates that save/restore-based
branching needs simulator state beyond what scene.get_state() exposes. It does
not identify which unexposed PhysX or task state causes the divergence.

--------------------------------------------------------------------------------
HYPOTHESIS UNDER TEST
    Let S = scene.get_state() at some step. Restoring that exact state with
    scene.reset_to(S) would be replay-equivalent if continuing both branches with
    identical recorded action tensors produced an identical trajectory. The
    public API may promise visible-state restoration rather than this stronger
    contract; the purpose of this reproducer is to make that distinction testable.

OBSERVED behaviour
    After the policy has moved toward the cube, the two continuations diverge
    despite get_state() reporting an exact round trip. A control run with
    --grasp_steps 0 remains identical for all compared post-step observations in
    the tested environment. The effect therefore depends on evolved task or
    simulator state, but contact-specific causality remains unproven.

    Save/restore-based counterfactual analysis therefore cannot assume replay
    equivalence without an explicit trajectory-level validation. Whether this is
    expected API behavior or missing state is a question for maintainers.

EXACT ENVIRONMENT (fill in yours when filing)
    Isaac Lab 4.5.22 ; Isaac Sim 6.0 ; task Isaac-Lift-Cube-Franka-v0
    Isaac 4.5 task assets ; PhysX GPU pipeline ; num_envs=1 ; single CUDA GPU
    Ubuntu 22.04 ; Python 3.12 ; rsl-rl-lib 5.0.1

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
parser.add_argument(
    "--asset_root",
    help="Optional Isaac asset root override applied before task registration.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

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


def load_published_policy(runner, checkpoint: str, device: str) -> None:
    """Load current or legacy Isaac Lab RSL-RL checkpoints strictly."""
    try:
        payload = torch.load(checkpoint, weights_only=True, map_location=device)
    except TypeError:
        raise RuntimeError("This reproducer requires torch with weights_only checkpoint loading.") from None
    except Exception as exc:
        raise RuntimeError("Checkpoint was rejected by safe tensor-only loading.") from exc
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")

    actor = runner.alg.get_policy()
    current = payload.get("actor_state_dict")
    if current is not None:
        if not isinstance(current, dict):
            raise TypeError("checkpoint actor_state_dict must be a dictionary")
        actor.load_state_dict(current, strict=True)
        return

    legacy = payload.get("model_state_dict")
    if not isinstance(legacy, dict):
        raise KeyError(
            "Checkpoint has neither 'actor_state_dict' nor legacy 'model_state_dict': "
            f"{sorted(payload)}"
        )

    mapped = {}
    for target_key, target_value in actor.state_dict().items():
        if target_key.startswith("mlp."):
            source_key = f"actor.{target_key.removeprefix('mlp.')}"
        elif target_key == "distribution.std_param":
            source_key = "std"
        else:
            raise KeyError(f"No legacy checkpoint mapping for actor key: {target_key}")
        if source_key not in legacy:
            raise KeyError(f"Legacy checkpoint is missing actor key: {source_key}")
        if legacy[source_key].shape != target_value.shape:
            raise ValueError(
                f"Shape mismatch for {source_key} -> {target_key}: "
                f"checkpoint {tuple(legacy[source_key].shape)}, actor {tuple(target_value.shape)}"
            )
        mapped[target_key] = legacy[source_key]

    actor.load_state_dict(mapped, strict=True)
    print("Loaded published legacy RSL-RL checkpoint through strict actor-only mapping.")


def main() -> None:
    torch.manual_seed(args.seed)
    if args.asset_root:
        import isaaclab.utils.assets as assets

        root = args.asset_root.rstrip("/")
        assets.NUCLEUS_ASSET_ROOT_DIR = root
        assets.NVIDIA_NUCLEUS_DIR = f"{root}/NVIDIA"
        assets.ISAAC_NUCLEUS_DIR = f"{root}/Isaac"
        assets.ISAACLAB_NUCLEUS_DIR = f"{root}/Isaac/IsaacLab"

        # Some runtime import paths load the robot configs before this standalone
        # script can apply the module-level asset constants.
        import isaaclab_assets.robots.franka as franka  # noqa: PLC0415

        panda_path = f"{assets.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
        franka.FRANKA_PANDA_CFG.spawn.usd_path = panda_path
        franka.FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path = panda_path

    import isaaclab_tasks  # noqa: F401, PLC0415
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: PLC0415

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
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
    load_published_policy(runner, ckpt, str(dev))
    policy = runner.get_inference_policy(device=str(dev))

    def act(o):
        with torch.inference_mode():
            return policy(o).clone()

    scene = env.unwrapped.scene
    obs = env.get_observations()
    for _ in range(args.grasp_steps):  # advance the policy toward the cube
        obs, *_ = env.step(act(obs))

    S = _clone(scene.get_state())

    snapshot_obs = obs_vec(obs)

    # (A) Natural continuation from the exposed state, recording the exact
    # action tensors that branch B will replay after reset_to.
    obs_a, traj_a, recorded_actions = obs, [], []
    for _ in range(args.compare_steps):
        action = act(obs_a)
        recorded_actions.append(action.clone())
        obs_a, *_ = env.step(action)
        traj_a.append(obs_vec(obs_a))

    # Restore the EXACT state we snapshotted, then verify get_state() round-trips it.
    scene.reset_to(S, torch.tensor([0], device=dev))
    round_trip = state_maxabs_diff(S, _clone(scene.get_state()))

    # (B) Continuation after reset_to(S) -- identical actions, identical state.
    obs_b, traj_b = env.get_observations(), []
    restored_obs_gap = float(np.abs(snapshot_obs - obs_vec(obs_b)).max())
    for action in recorded_actions:
        obs_b, *_ = env.step(action.clone())
        traj_b.append(obs_vec(obs_b))

    gaps = [float(np.abs(a - b).max()) for a, b in zip(traj_a, traj_b, strict=False)]

    print("\n================ Isaac Lab reset_to contact-state MRE ================")
    print(f"task                         : {args.task}  (num_envs=1)")
    print(f"visible state round-trip     : max|get_state()-S| = {round_trip:.2e}  (expected ~0)")
    print(f"pre-step restored obs gap    : {restored_obs_gap:.2e}")
    print(f"per-step max obs gap A vs B  : {[f'{g:.2e}' for g in gaps]}")
    print(f"final obs gap (step {args.compare_steps:2d})       : {gaps[-1]:.3e}")
    reproduced = round_trip < 1e-4 and gaps[-1] > 1e-2
    print("\nHYPOTHESIS: gaps are ~0 if reset_to(get_state()) is replay-equivalent.")
    if reproduced:
        print("OBSERVED: gaps grow despite an exact exposed-state round trip.")
        print("          The missing simulator/task state is not identified by this MRE.")
    else:
        print("OBSERVED: no post-step trajectory divergence at this warm-up length.")
    print(
        "\nRESULT: "
        f"{'REPLAY DIVERGENCE REPRODUCED' if reproduced else 'not reproduced (adjust grasp_steps?)'}"
    )

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
