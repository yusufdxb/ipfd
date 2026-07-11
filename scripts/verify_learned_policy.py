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


def make_on_step(env, z_rest):
    """Failure-injection hook for the library rollout: once the cube is genuinely
    lifted, either force the gripper open (recoverable slip) or teleport the cube
    out of reach (irrecoverable). Mirrors the previously verified standalone driver."""
    trig = {"on": False}

    def on_step(step, e, actions):
        if not trig["on"] and obj_z(e, 0) > z_rest + args.lift_margin:
            trig["on"] = True
            if args.failure == "teleport":
                teleport_out_of_reach(e, args.reach_push)
        if trig["on"] and args.failure == "slip":
            force_gripper_open(actions, 0, args.gripper_open)

    return on_step


def main() -> None:
    from ipfd.adapters.isaac_lab import collect_rollout

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

    # Drive the PACKAGED library API end-to-end (env-isolated probe + PoNR live here).
    env.reset()
    rollout = collect_rollout(
        env, policy,
        object_height=obj_z, rest_height=z_rest, lift_threshold=args.lift_margin,
        recovery_policy=policy if args.probe else None,
        max_steps=args.max_steps, probe_stride=args.probe_stride, probe_budget=args.probe_budget,
        on_step=make_on_step(env, z_rest), seed=args.seed,
        meta={"policy": "rsl_rl_ppo", "checkpoint": os.path.basename(args.checkpoint)},
    )
    log(f"primary: T={rollout.T} t_failure={rollout.t_failure} success={rollout.success} "
        f"probe_resets={rollout.meta.get('probe_resets')} "
        f"primary_integrity_max_delta={rollout.meta.get('primary_integrity_max_delta'):.2e} m")

    report = build_report(rollout)
    print(report.summary())

    ent = rollout.entropy
    ent_flat = ent is None or (ent.size and float(np.std(ent)) < 1e-6)
    print("\n=== IPFD_LEARNED_STATUS ===")
    print("real_learned_policy: YES")
    print(f"detector_alarm_fired: {'YES' if report.t_alarm is not None else 'NO'}")
    print(f"entropy_signal: {'FLAT (state-independent std)' if ent_flat else 'VARIES'}")
    print(f"ponr_detected: {'YES' if report.t_ponr is not None else 'NO'}")
    print(f"primary_integrity_max_delta_m: {rollout.meta.get('primary_integrity_max_delta')}")
    print(f"failure_lead_time_s: {report.failure_lead_time_s}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
