"""Phase 3: run IPFD on a REAL trained (learned) manipulation policy.

Unlike the scripted-state-machine evidence chain, this drives IPFD with a policy
trained by Isaac Lab's own ``rsl_rl`` PPO on ``Isaac-Lift-Cube-Franka-v0`` (joint
space). It answers the question Phase 3 exists to answer:

    On a real learned policy, do IPFD's detectors (action-variance / entropy /
    embedding-drift) fire, and does the recovery-probe PoNR localise a real
    silent failure -- or not? Report the measurement, whatever it is.

Failure model (a genuine silent-doom window, not an instantaneous teleport):
once the policy has LIFTED the cube, we physically force env 0's gripper OPEN for
the rest of the episode. The arm keeps executing the policy's lift command (looks
fine), but the cube slips and falls under gravity; ``object_dropping`` becomes
externally observable only when it has fallen far enough. The interval between the
slip and that observable drop is the silent window IPFD should expose.

Two passes, using the env-isolation mechanic proven in ``verify_pnor_isolated.py``
/ ``verify_pnor_grasped.py`` (env 0 = pristine primary, never ``reset_to``; env 1 =
probe cell):

  Pass 1  record the primary rollout in env 0 with the learned policy, capturing
          per-step action + entropy + penultimate embedding (via
          :class:`ipfd.oracles.rsl_rl_policy.LearnedPolicy`) and per-step saved
          sim state. Inject the gripper slip once lifted.
  Pass 2  (--probe) for each saved checkpoint, offset it into env 1, ``reset_to``
          env 1 only, and run the SAME learned policy there as the recovery
          oracle for a fixed budget: recovery_success[t] = did it re-lift?

Then build the IPFD report and print an ``IPFD_LEARNED_STATUS`` block. Honest by
construction: if a detector has no signal (e.g. a state-independent action std
makes entropy constant), the report shows it flat rather than hiding it.

Run:
    OMNI_KIT_ACCEPT_EULA=YES ~/Sim/isaac-sim-venv/bin/python \\
        scripts/verify_learned_policy.py --headless --checkpoint <model_*.pt> --probe
"""

from __future__ import annotations

# ruff: noqa: I001, E402
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD on a trained rsl_rl policy")
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--checkpoint", required=True, help="Path to an rsl_rl model_*.pt")
parser.add_argument("--num_envs", type=int, default=4, help=">=2 for the isolation probe")
parser.add_argument("--max_steps", type=int, default=220)
parser.add_argument("--lift_margin", type=float, default=0.06, help="Object rise [m] counted as lifted.")
parser.add_argument("--gripper_open", type=float, default=1.0, help="Forced gripper-open action value.")
parser.add_argument("--failure", choices=["slip", "teleport"], default="slip",
                    help="slip = recoverable gripper-open; teleport = irrecoverable out-of-reach.")
parser.add_argument("--reach_push", type=float, default=1.0, help="Out-of-reach displacement [m] (teleport).")
parser.add_argument("--probe", action="store_true", help="Run the env-isolated recovery-probe PoNR.")
parser.add_argument("--probe_stride", type=int, default=8)
parser.add_argument("--probe_budget", type=int, default=90)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from ipfd import build_report
from ipfd.types import Rollout
from ipfd.adapters.isaac_lab import slice_state, offset_root_positions
from ipfd.oracles.rsl_rl_policy import load_learned_policy


def log(msg: str) -> None:
    print(f"[learned] {msg}", flush=True)


def obj_z(env, i: int) -> float:
    origin_z = float(env.unwrapped.scene.env_origins[i, 2].item())
    pos = wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)
    return float(pos[i, 2].item()) - origin_z


def rest_height(env, i: int, n_settle: int = 12) -> float:
    zs = [obj_z(env, i) for _ in range(n_settle)]
    return float(np.min(zs))


def force_gripper_open(actions: torch.Tensor, env_i: int, value: float) -> None:
    actions[env_i, -1] = value


def teleport_out_of_reach(env, push: float) -> None:
    """Irrecoverable doom: shove env-0's cube +x out of the arm's reach (once)."""
    obj = env.unwrapped.scene["object"]
    pose = wp.to_torch(obj.data.root_pose_w)[0:1].clone()
    pose[0, 0] += push
    obj.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.unwrapped.device))


def record_primary(env, policy, dt, z_rest, args):
    """Pass 1: pristine env-0 rollout with the learned policy + injected slip."""
    obs = env.get_observations()
    obs_list, act_list, ent_list, emb_list = [], [], [], []
    states: dict[int, object] = {}
    slip_step = None
    t_failure = None
    success = False

    for step in range(args.max_steps):
        if step % args.probe_stride == 0:
            states[step] = slice_state(env.unwrapped.scene.get_state(), slice(0, 1))

        actions = policy(obs)  # captures embedding + entropy for all envs
        obs_vec = obs["policy"][0].detach().float().cpu().numpy().reshape(-1)
        act_vec = actions[0].detach().float().cpu().numpy().reshape(-1)
        emb = policy.last_embedding[0] if policy.last_embedding is not None else None
        ent = float(policy.last_entropy[0]) if policy.last_entropy is not None else None

        z = obj_z(env, 0)
        if slip_step is None and z > z_rest + args.lift_margin:
            slip_step = step  # first genuine lift -> inject the failure here
            if args.failure == "teleport":
                teleport_out_of_reach(env, args.reach_push)
        if slip_step is not None and args.failure == "slip":
            force_gripper_open(actions, 0, args.gripper_open)

        obs_list.append(obs_vec)
        act_list.append(act_vec)
        if ent is not None:
            ent_list.append(ent)
        if emb is not None:
            emb_list.append(emb)

        obs, _rew, dones, _extra = env.step(actions)
        policy.reset(dones)

        if slip_step is not None and bool(dones[0].item()):
            t_failure = step  # object_dropping / timeout became observable
            break

    T = len(obs_list)
    if t_failure is None:
        success = slip_step is not None and obj_z(env, 0) > z_rest + args.lift_margin
        if not success:
            t_failure = T - 1

    return {
        "obs": np.asarray(obs_list),
        "act": np.asarray(act_list),
        "ent": np.asarray(ent_list) if len(ent_list) == T else None,
        "emb": np.asarray(emb_list) if len(emb_list) == T else None,
        "states": states,
        "slip_step": slip_step,
        "t_failure": t_failure,
        "success": success,
        "T": T,
    }


def evaluate_recovery(env, policy, states, z_rest, args) -> dict[int, bool]:
    """Pass 2: env-isolated learned-oracle recovery from each saved checkpoint."""
    origins = env.unwrapped.scene.env_origins
    delta = (origins[1] - origins[0]).detach()
    verdicts: dict[int, bool] = {}
    loc_max = 0.0

    for step, state in sorted(states.items()):
        pose_before = wp.to_torch(env.unwrapped.scene["object"].data.root_pose_w)[0].detach().clone()
        state_probe = offset_root_positions(state, delta)
        env_ids = torch.tensor([1], device=env.unwrapped.device, dtype=torch.long)
        env.unwrapped.scene.reset_to(state_probe, env_ids)
        pose_after = wp.to_torch(env.unwrapped.scene["object"].data.root_pose_w)[0].detach()
        loc_max = max(loc_max, float((pose_after - pose_before).abs().max().item()))
        if hasattr(env.unwrapped, "episode_length_buf"):
            env.unwrapped.episode_length_buf[:] = 0

        obs = env.get_observations()
        recovered = False
        for _ in range(args.probe_budget):
            actions = policy(obs)
            obs, _rew, dones, _extra = env.step(actions)
            if obj_z(env, 1) > z_rest + args.lift_margin:
                recovered = True
                break
            if bool(dones[1].item()):
                break
        verdicts[step] = recovered

    log(f"probe primary-integrity max env-0 pose delta = {loc_max:.2e} m across {len(states)} resets")
    return verdicts


def forward_fill(verdicts: dict[int, bool], T: int) -> np.ndarray:
    from ipfd.adapters.isaac_lab import forward_fill_recovery
    return forward_fill_recovery(verdicts, T)


def main() -> None:
    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    device = str(env.unwrapped.device)
    log(f"env {args.task} num_envs={env.num_envs} device={device}")

    policy = load_learned_policy(env, agent_cfg.to_dict(), args.checkpoint, device=device)
    log(f"loaded policy from {args.checkpoint}")

    env.reset()
    for _ in range(15):  # let the scene settle
        env.step(torch.zeros((env.num_envs, env.action_space.shape[1]), device=device))
    z_rest = rest_height(env, 0)
    log(f"settled object rest height (env0) = {z_rest:.3f} m")

    env.reset()
    prim = record_primary(env, policy, float(env.unwrapped.step_dt), z_rest, args)
    log(f"primary: T={prim['T']} slip_step={prim['slip_step']} "
        f"t_failure={prim['t_failure']} success={prim['success']}")

    recovery = None
    if args.probe and prim["states"]:
        try:
            env.reset()
            verdicts = evaluate_recovery(env, policy, prim["states"], z_rest, args)
            recovery = forward_fill(verdicts, prim["T"])
            log(f"recovery verdicts: {sorted(verdicts.items())}")
        except Exception as exc:  # pragma: no cover
            log(f"PROBE FAILED (reporting detectors only): {type(exc).__name__}: {exc}")

    rollout = Rollout(
        observations=prim["obs"],
        actions=prim["act"],
        entropy=prim["ent"],
        embeddings=prim["emb"],
        success=prim["success"],
        t_failure=None if prim["success"] else prim["t_failure"],
        recovery_success=recovery,
        dt=float(env.unwrapped.step_dt),
        seed=args.seed,
        meta={"source": "isaac_lab", "policy": "rsl_rl_ppo", "task": args.task,
              "checkpoint": os.path.basename(args.checkpoint)},
    )
    report = build_report(rollout)
    print(report.summary())

    ent = rollout.entropy
    ent_flat = ent is None or (ent.size and float(np.std(ent)) < 1e-6)
    print("\n=== IPFD_LEARNED_STATUS ===")
    print(f"real_learned_policy: YES")
    print(f"detector_alarm_fired: {'YES' if report.t_alarm is not None else 'NO'}")
    print(f"entropy_signal: {'FLAT (state-independent std)' if ent_flat else 'VARIES'}")
    print(f"ponr_detected: {'YES' if report.t_ponr is not None else 'NO'}")
    if report.t_ponr is not None and report.t_failure is not None:
        print(f"silent_doom_window_s: {report.silent_doom_window_s}")
    print(f"failure_lead_time_s: {report.failure_lead_time_s}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
