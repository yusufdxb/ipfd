"""Root-cause: is an interleaved recovery probe TRANSPARENT to the primary rollout?

IPFD's in-loop PoNR needs one property: injecting a recovery probe (save state ->
run a fresh scripted grasp+lift for K steps -> reset_to) must leave the primary
env evolving EXACTLY as if no probe had happened. ``verify_state_fidelity.py``
already proved a SINGLE-step reset_to is bit-exact; ``verify_real_policy.py``
showed the full in-loop probe corrupts the primary. This script isolates WHY.

Hypothesis
----------
Single-step reset_to is bit-exact, but reset_to captures only KINEMATIC state
(root/joint pose+velocity) -- NOT the PhysX solver's contact-manifold / warm-start
cache. So restoring AFTER a probe has established a grasp is lossy, and the loss
grows with contact. Prediction: a probe injected BEFORE the arm contacts the cube
is transparent (A==B); a probe injected AFTER a grasp is not.

Method (single env, real physics)
---------------------------------
From a common saved state S0 reached at a chosen grasp stage:
  * Branch A (baseline): run the primary SM N steps, record obs + object height.
  * Restore S0 + SM progress, then Branch B (probed): inject ONE probe
    (get_state -> fresh SM K steps -> reset_to), then run the SAME N steps.
  * Compare A vs B: per-step max-abs obs diff, divergence onset, final height gap.
Also report the probe's own multi-step WRITE-BACK diff (entities just before the
probe vs immediately after its reset_to) -- lossy write-back is the direct cause.

Two conditions: probe injected PRE-CONTACT vs POST-GRASP. Prints a machine block.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_probe_transparency.py --headless
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD probe-transparency root-cause")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--compare_steps", type=int, default=40, help="N steps to compare A vs B.")
parser.add_argument("--probe_budget", type=int, default=120, help="Steps the injected probe runs.")
parser.add_argument("--tol", type=float, default=1e-4, help="Max-abs-diff transparency tolerance.")
parser.add_argument("--seed", type=int, default=1)
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
sys.path.insert(0, _HERE)
wp.init()
from _lift_sm import (  # noqa: E402
    PickAndLiftSm, sm_action, object_z, identity_action,
)

# SM state ids (from _lift_sm.PickSmState): 0 REST, 1 APPROACH_ABOVE, 2 APPROACH,
# 3 GRASP, 4 LIFT. Used as plain ints to avoid warp-version .val attribute risk.
STATE_APPROACH_ABOVE = 1
STATE_LIFT = 4

RESULTS: dict[str, str] = {}
NOTES: list[str] = []


def log(msg: str) -> None:
    print(f"[transparency] {msg}", flush=True)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max().item())


def snapshot_entities(env) -> dict[str, torch.Tensor]:
    scene = env.unwrapped.scene
    robot = scene["robot"].data
    obj = scene["object"].data
    return {
        "joint_pos": wp.to_torch(robot.joint_pos).clone(),
        "joint_vel": wp.to_torch(robot.joint_vel).clone(),
        "obj_pos": wp.to_torch(obj.root_pos_w).clone(),
        "obj_quat": wp.to_torch(obj.root_quat_w).clone(),
        "obj_vel": wp.to_torch(obj.root_vel_w).clone(),
    }


def entities_maxdiff(p: dict, q: dict) -> float:
    return max(maxdiff(p[k], q[k]) for k in p)


def drive_to_state(env, sm, des_or, target_state: int, cap: int) -> int:
    """Step the primary SM until sm_state reaches target_state (or cap). Returns steps used."""
    action = identity_action(env.unwrapped.device)
    for i in range(cap):
        env.step(action)
        if int(sm.sm_state[0].item()) >= target_state:
            return i + 1
        action = sm_action(env, sm, des_or)
    return cap


def inject_probe(env, des_or, dt, dev) -> float:
    """One recovery probe. Returns the multi-step write-back diff (pre vs post reset_to)."""
    S = env.unwrapped.scene.get_state()
    p_pre = snapshot_entities(env)
    ep = env.unwrapped.episode_length_buf.clone() if hasattr(env.unwrapped, "episode_length_buf") else None
    psm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)
    action = sm_action(env, psm, des_or)
    try:
        for _ in range(args.probe_budget):
            env.step(action)
            action = sm_action(env, psm, des_or)
    finally:
        env.unwrapped.scene.reset_to(S, None)
        if ep is not None:
            env.unwrapped.episode_length_buf[:] = ep
    p_post = snapshot_entities(env)
    return entities_maxdiff(p_pre, p_post)


def run_branch(env, sm, des_or, s0, sm_snap, n, dev, probe=False, dt=None):
    """Restore to (s0, sm_snap), optionally inject a probe, then run n primary steps."""
    env.unwrapped.scene.reset_to(s0, None)
    sm.restore(sm_snap)
    wb = None
    if probe:
        wb = inject_probe(env, des_or, dt, dev)
    obs_traj, z_traj = [], []
    action = sm_action(env, sm, des_or)  # first action from the common restored state
    for _ in range(n):
        obs, _r, term, trunc, _i = env.step(action)
        obs_traj.append(obs["policy"].detach().cpu().double().reshape(-1))
        z_traj.append(object_z(env))
        action = sm_action(env, sm, des_or)
        if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
            break
    return torch.stack(obs_traj), np.asarray(z_traj), wb


def run_condition(env, label: str, target_state: int, dt: float, dev) -> None:
    des_or = torch.zeros((1, 4), device=dev)
    des_or[:, 1] = 1.0
    env.reset(seed=args.seed)
    env.step(identity_action(dev))
    sm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)
    used = drive_to_state(env, sm, des_or, target_state, cap=140)
    reached = int(sm.sm_state[0].item())
    log(f"[{label}] drove {used} steps -> sm_state={reached} (target {target_state}), z={object_z(env):.3f}")

    # Common branch point.
    s0 = env.unwrapped.scene.get_state()
    sm_snap = sm.snapshot()
    n = args.compare_steps

    obs_a, z_a, _ = run_branch(env, sm, des_or, s0, sm_snap, n, dev, probe=False)
    obs_b, z_b, wb = run_branch(env, sm, des_or, s0, sm_snap, n, dev, probe=True, dt=dt)

    m = min(obs_a.shape[0], obs_b.shape[0])
    per_step = (obs_a[:m] - obs_b[:m]).abs().amax(dim=1).numpy()
    obs_div = float(per_step.max()) if m else float("nan")
    onset = next((int(i) for i, d in enumerate(per_step) if d > args.tol), -1)
    z_gap = float(abs(z_a[min(m, len(z_a)) - 1] - z_b[min(m, len(z_b)) - 1])) if m else float("nan")

    transparent = obs_div <= args.tol
    log(f"[{label}] probe multi-step WRITE-BACK diff (pre vs post reset_to) = {wb:.3e}")
    log(f"[{label}] A-vs-B obs max-abs-diff = {obs_div:.3e}  divergence onset step = {onset}  "
        f"final object-height gap = {z_gap:.3e}")
    log(f"[{label}] probe transparent = {transparent} (tol {args.tol:.0e})")
    RESULTS[f"{label}_writeback_diff"] = f"{wb:.3e}"
    RESULTS[f"{label}_ab_obs_diff"] = f"{obs_div:.3e}"
    RESULTS[f"{label}_transparent"] = "YES" if transparent else "NO"


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=1)
        env = gym.make(args.env_id, cfg=env_cfg)
        dev = env.unwrapped.device
        dt = env_cfg.sim.dt * env_cfg.decimation
        log(f"env={args.env_id} dt={dt:.4f} act_dim={env.action_space.shape[-1]}")

        # Condition 1: probe injected BEFORE contact (arm still approaching above).
        run_condition(env, "pre_contact", target_state=STATE_APPROACH_ABOVE, dt=dt, dev=dev)
        # Condition 2: probe injected AFTER the grasp closes (contact established).
        run_condition(env, "post_grasp", target_state=STATE_LIFT, dt=dt, dev=dev)

        pre_ok = RESULTS.get("pre_contact_transparent") == "YES"
        post_ok = RESULTS.get("post_grasp_transparent") == "YES"
        if pre_ok and not post_ok:
            RESULTS["root_cause"] = "CONTACT_STATE_NOT_RESTORED"
            NOTES.append("Pre-contact probe is transparent but post-grasp is not: reset_to does not "
                         "restore PhysX contact/solver state, so in-loop probing after a grasp is lossy.")
        elif pre_ok and post_ok:
            RESULTS["root_cause"] = "PROBE_TRANSPARENT_LOOK_ELSEWHERE"
            NOTES.append("Both probes are transparent: the real-rollout corruption is NOT the probe "
                         "restore itself; re-examine the primary-loop integration in verify_real_policy.py.")
        else:
            RESULTS["root_cause"] = "PROBE_NOT_TRANSPARENT_EVEN_PRE_CONTACT"
            NOTES.append("Even the pre-contact probe corrupts the primary: the loss is not grasp-specific.")
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
        if NOTES:
            print("NOTES:")
            for nline in NOTES:
                print(f"  - {nline}")
            print("-" * 55)
        print("PROBE_TRANSPARENCY:")
        for k in ("pre_contact_writeback_diff", "pre_contact_ab_obs_diff", "pre_contact_transparent",
                  "post_grasp_writeback_diff", "post_grasp_ab_obs_diff", "post_grasp_transparent",
                  "root_cause"):
            if k in RESULTS:
                print(f"- {k}: {RESULTS[k]}")
        print("=" * 55, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
