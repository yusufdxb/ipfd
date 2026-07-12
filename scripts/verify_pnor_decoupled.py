"""Meaningful PoNR under a real policy via a DECOUPLED recovery probe.

Root cause (see verify_probe_transparency.py): reset_to restores kinematics
bit-exactly but NOT PhysX contact state, so probing *inside* a live rollout
corrupts the primary once a grasp is established. The fix needs no second env:
DECOUPLE probing from the primary.

  Pass 1 (record): run the primary policy once, uninterrupted. Save a full
    scene state S_t and record (obs, action, object height) at every step.
    Because nothing is restored during this pass, the recorded trajectory is
    uncorrupted -- it is the real rollout.

  Pass 2 (evaluate): for a strided set of steps t, reset_to(S_t) and run a FRESH
    scripted pick-lift for a budget; recovery_success[t] = did it lift? These
    passes DO leave a cold contact cache, but we never resume the primary, so
    there is nothing left to corrupt. Each probe restores a fresh checkpoint
    (kinematics bit-exact) and re-grasps from scratch, so its verdict is faithful.

PoNR = first step after which recovery never again succeeds. With the cube
displaced out of reach at a KNOWN step, recovery_success must flip True->False
there, and IPFD's PoNR must land there -- NOT at step 0.

A determinism check (same checkpoint probed twice -> same verdict) guards the
"cold contact cache makes the verdict unfaithful" risk.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_pnor_decoupled.py --headless

RESULT (2026-07-05, Isaac Lab 4.5.22, an NVIDIA Blackwell-class consumer GPU) -- honest, negative:

  The decoupling is necessary but NOT sufficient, because of a deeper, directly
  confirmed Isaac Lab behaviour: ``reset_to_poisons_env: YES``. A fresh episode
  lifts the cube (CONTROL zmax 0.341). After exactly ONE reset_to probe followed
  by ``env.reset(seed=0)``, the SAME seed no longer lifts (POISON TEST zmax
  0.045). So a single reset_to() permanently corrupts the PhysX sim and the
  corruption SURVIVES env.reset(). Consequently pass-2 recovery probes poison the
  shared env for each other, the re-recorded rollouts are degenerate, and the
  recovery_success verdicts are invalid (recover T/F counts here are spurious).
  meaningful_pnor_detected = NO.

  Conclusion: probe-based PoNR cannot share a single env instance in this Isaac
  Lab version. A correct implementation needs env ISOLATION (a separate sim
  instance per probe / a probe pool) or a reset_to that also restores PhysX
  solver+contact state. Single-step state restore is still bit-exact
  (verify_state_fidelity.py); the block is the persistent reset_to side effect,
  measured here against ground truth.
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD decoupled-probe PoNR")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--max_steps", type=int, default=150)
parser.add_argument("--perturb_step", type=int, default=60)
parser.add_argument("--probe_stride", type=int, default=8, help="Probe every N recorded steps.")
parser.add_argument("--probe_budget", type=int, default=140, help="Steps a fresh SM gets to recover.")
parser.add_argument("--lift_thresh", type=float, default=0.04, help="Object rise [m] counted as a lift.")
parser.add_argument("--reach_push", type=float, default=1.2, help="Out-of-reach displacement [m] in +x.")
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
sys.path.insert(0, os.path.join(_REPO, "src"))
wp.init()
from ipfd.oracles.pick_lift_sm import PickAndLiftSm, sm_action, object_z, identity_action  # noqa: E402
from ipfd import build_report  # noqa: E402
from ipfd.types import Rollout  # noqa: E402
from ipfd.ponr import point_of_no_return  # noqa: E402


def log(msg: str) -> None:
    print(f"[decoupled] {msg}", flush=True)


def clone_state(state):
    """Deep-clone a scene state dict so per-step checkpoints do not alias."""
    if torch.is_tensor(state):
        return state.clone()
    if isinstance(state, dict):
        return {k: clone_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(clone_state(v) for v in state)
    return state


def displace_object_out_of_reach(env, push: float) -> None:
    obj = env.unwrapped.scene["object"]
    pose = wp.to_torch(obj.data.root_pose_w).clone()
    pose[:, 0] += push
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(wp.to_torch(obj.data.root_vel_w).clone() * 0.0)


def record_primary(env, dt, dev, seed, perturb_step):
    """Pass 1: run the policy once, uninterrupted; checkpoint every step."""
    des_or = torch.zeros((1, 4), device=dev)
    des_or[:, 1] = 1.0
    env.reset(seed=seed)
    env.step(identity_action(dev))
    sm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)

    obs_list, act_list, objz, states = [], [], [], []
    action = identity_action(dev)
    for step in range(args.max_steps):
        if perturb_step is not None and step == perturb_step:
            displace_object_out_of_reach(env, args.reach_push)
        obs, _r, term, trunc, _i = env.step(action)
        obs_list.append(obs["policy"].detach().cpu().numpy().reshape(-1).astype(np.float64))
        act_list.append(action.detach().cpu().numpy().reshape(-1).astype(np.float64))
        objz.append(object_z(env))
        states.append(clone_state(env.unwrapped.scene.get_state()))
        action = sm_action(env, sm, des_or)
        if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
            break
    # Settled rest height: the cube spawns above the table and drops; the SM's
    # REST phase holds the arm still for the first ~10 steps, so the minimum over
    # the early window is the true resting height (a one-step-post-reset read is
    # captured mid-fall and over-estimates it, inflating the lift threshold).
    z0 = float(np.min(objz[: min(15, len(objz))]))
    return obs_list, act_list, objz, states, z0, des_or


def probe_from(env, state, des_or, z0, dt, dev) -> bool:
    """Pass 2 unit: reset_to a checkpoint, run a FRESH pick-lift, True if it lifts."""
    env.unwrapped.scene.reset_to(clone_state(state), None)
    sm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)
    action = sm_action(env, sm, des_or)
    for _ in range(args.probe_budget):
        env.step(action)
        if object_z(env) > z0 + args.lift_thresh:
            return True
        action = sm_action(env, sm, des_or)
    return False


def evaluate_recovery(env, states, des_or, z0, dt, dev, perturb_step):
    """Pass 2: strided (dense near the doom) recovery probes; forward-fill between."""
    T = len(states)
    probe_steps = set(range(0, T, args.probe_stride))
    probe_steps.add(T - 1)
    if perturb_step is not None:
        probe_steps.update(t for t in range(max(0, perturb_step - 4), min(T, perturb_step + 12)))
    probe_steps = sorted(probe_steps)

    verdict = {}
    for t in probe_steps:
        verdict[t] = probe_from(env, states[t], des_or, z0, dt, dev)
    # forward-fill to every step (recoverability treated as piecewise-constant between probes)
    rec = np.zeros(T, dtype=bool)
    last = True
    for t in range(T):
        if t in verdict:
            last = verdict[t]
        rec[t] = last
    return rec, probe_steps, verdict


def run_case(env, name, dt, dev, seed, perturb_step, status_acc):
    obs_list, act_list, objz, states, z0, des_or = record_primary(env, dt, dev, seed, perturb_step)
    T = len(states)
    rec, probe_steps, verdict = evaluate_recovery(env, states, des_or, z0, dt, dev, perturb_step)

    lifted_end = objz[-1] > z0 + args.lift_thresh
    success = lifted_end and perturb_step is None
    rollout = Rollout(
        observations=np.asarray(obs_list), actions=np.asarray(act_list),
        entropy=None, embeddings=None, success=success,
        t_failure=None if success else T - 1,
        recovery_success=rec, dt=dt, seed=seed,
        meta={"source": "isaac_lab", "robot": "franka", "task": "lift",
              "policy": "scripted_pick_lift", "perturb_step": perturb_step, "decoupled": True},
    )
    report = build_report(rollout)
    ponr = point_of_no_return(rec)
    n_true, n_false = int(rec.sum()), int((~rec).sum())
    zmax = float(np.max(objz))
    log(f"--- {name}: T={T} z0={z0:.3f} zmax={zmax:.3f} z_end={objz[-1]:.3f} "
        f"lifted_primary={zmax > z0 + args.lift_thresh} lifted_end={lifted_end} "
        f"success={success} recover(T/F)={n_true}/{n_false} probes={len(probe_steps)}")
    log("    raw probe verdicts: " + ", ".join(f"{t}:{'T' if verdict[t] else 'F'}" for t in probe_steps))
    # show the True->False flip in the raw probe verdicts
    flips = [t for t in probe_steps if verdict[t] is False]
    first_false = flips[0] if flips else None
    log(f"    PoNR={ponr} first_probe_False={first_false} injected_doom={perturb_step} "
        f"alarm={report.t_alarm} t_failure={rollout.t_failure}")

    status_acc["any_true"] |= n_true > 0
    status_acc["any_false"] |= n_false > 0
    if perturb_step is not None and ponr is not None and ponr > 0:
        status_acc["meaningful_ponr"] = True
        near = abs(ponr - perturb_step) <= args.probe_stride
        status_acc["ponr_near_doom"] |= near
        log(f"    -> PoNR {ponr} vs injected doom {perturb_step} (near={near}, tol={args.probe_stride})")
    if report.t_alarm is not None and rollout.t_failure is not None and rollout.t_failure > report.t_alarm:
        status_acc["lead_time"] = True


def main() -> None:
    env = None
    status = {"state_restore_fidelity": "PASS (verify_state_fidelity.py)",
              "reset_to_poisons_env": "UNKNOWN",
              "meaningful_pnor_detected": "NO", "recovery_oracle_non_degenerate": "NO",
              "measurable_failure_lead_time": "NO", "probe_verdict_deterministic": "UNKNOWN",
              "overall_status": "PARTIALLY_VERIFIED"}
    acc = {"any_true": False, "any_false": False, "meaningful_ponr": False,
           "ponr_near_doom": False, "lead_time": False}
    try:
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=1)
        env = gym.make(args.env_id, cfg=env_cfg)
        dev = env.unwrapped.device
        dt = env_cfg.sim.dt * env_cfg.decimation
        log(f"env={args.env_id} dt={dt:.4f} act_dim={env.action_space.shape[-1]}")

        # CONTROL (fresh env, FIRST episode, no prior probing): record_primary
        # calls get_state() every step -- does the SM still lift? Isolates a
        # get_state side-effect from reset_to corruption.
        _o, _a, objz_c, states_c, z0_c, des_c = record_primary(env, dt, dev, seed=0, perturb_step=None)
        control_lift = max(objz_c) > z0_c + args.lift_thresh
        log(f"CONTROL fresh record_primary(seed=0): z0={z0_c:.3f} zmax={max(objz_c):.3f} lifted={control_lift}")

        # POISON TEST: run exactly ONE reset_to probe, then env.reset() and record
        # the SAME seed again. If the lift is now gone, reset_to's contact/solver
        # corruption PERSISTS across env.reset() -- which breaks any probe-based
        # PoNR that shares one env instance, decoupled or not.
        tmid = min(len(states_c) - 1, int(0.6 * len(states_c)))
        _ = probe_from(env, states_c[tmid], des_c, z0_c, dt, dev)
        _o2, _a2, objz_c2, _s2, z0_c2, _d2 = record_primary(env, dt, dev, seed=0, perturb_step=None)
        post_lift = max(objz_c2) > z0_c2 + args.lift_thresh
        log(f"POISON TEST after ONE reset_to probe, record_primary(seed=0): "
            f"zmax={max(objz_c2):.3f} lifted={post_lift}")
        status["reset_to_poisons_env"] = "NO" if post_lift else "YES"
        status["probe_verdict_deterministic"] = "N/A"

        run_case(env, "nominal", dt, dev, seed=0, perturb_step=None, status_acc=acc)
        run_case(env, f"perturbed@{args.perturb_step}", dt, dev, seed=1,
                 perturb_step=args.perturb_step, status_acc=acc)

        status["meaningful_pnor_detected"] = "YES" if acc["meaningful_ponr"] else "NO"
        status["recovery_oracle_non_degenerate"] = "YES" if (acc["any_true"] and acc["any_false"]) else "NO"
        status["measurable_failure_lead_time"] = "YES" if acc["lead_time"] else "NO"
        core = acc["meaningful_ponr"] and acc["any_true"] and acc["any_false"] and acc["ponr_near_doom"]
        status["overall_status"] = "VERIFIED" if core else "PARTIALLY_VERIFIED"
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
        print("IPFD_DECOUPLED_PNOR_STATUS:")
        for k in ("state_restore_fidelity", "reset_to_poisons_env", "meaningful_pnor_detected",
                  "recovery_oracle_non_degenerate", "measurable_failure_lead_time", "overall_status"):
            print(f"- {k}: {status[k]}")
        print("=" * 55, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
