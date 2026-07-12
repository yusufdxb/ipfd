"""Dual-Environment Recovery Probe: measure PoNR WITHOUT disturbing the primary.

This is the terminal step in the recovery-probe line of evidence:

  * verify_state_fidelity.py     -> single-step reset_to is bit-exact.
  * verify_probe_transparency.py -> IN-LOOP probing corrupts the primary after a
                                    grasp (PhysX contact state not restored).
  * verify_pnor_decoupled.py     -> a single reset_to poisons a num_envs=1 sim
                                    even across env.reset() (global poison).
  * verify_multienv_isolation.py -> BUT per-env reset_to is LOCAL: churning env 1
                                    leaves env 0 bit-identical. => isolate probes.
  * verify_pnor_isolated.py      -> env-isolated probing works; primary pristine;
                                    residual gap: a PRE-GRASP failure needs the
                                    continue-oracle to re-approach from a cold
                                    PhysX contact state, which is unreliable.

ARCHITECTURE (the fix for the residual gap is architectural, not a threshold):

  Isaac Lab runs ONE SimulationContext per process, so two fully independent sim
  INSTANCES are not available in-process. The dual-environment probe is therefore
  a VECTORISED pool: env 0 is the PRIMARY and is NEVER reset_to or action-
  corrupted by the probe; envs 1..P are PROBE cells that receive exported
  snapshots and diverge freely. Per-env reset_to being local (proven above) is
  what makes the two rollouts independent.

    Primary env 0 ----(get_state, origin-shifted)----> Probe env 1
        |                                                   |
     pristine rollout                              recovery rollouts
     (never restored)                              (reset_to + continue)
        |                                                   |
     recorded once                                    recoverable?

  Failure model: a GRASPED-REGION gripper slip. The primary is driven to grasp
  and lift the cube, then its gripper is forced OPEN for a short window so the
  cube drops -- a failure that lands squarely in the region where the continue-
  oracle is reliable (it restores a LIFT-state snapshot and keeps lifting). This
  is NOT threshold/detector tuning: it places the injected doom inside the
  oracle's supported domain instead of the cold-contact pre-grasp region.

  Recovery oracle: for a strided set of primary checkpoints S_t, export S_t into
  the probe env (origin-shifted, since get_state poses are absolute), restore the
  primary's SM progress, and CONTINUE the policy for a budget. recovery[t] = did
  the probe cell finish the lift? Checkpoints BEFORE the slip still hold the cube
  -> recover (True). Checkpoints after the cube separates -> cannot re-grasp mid-
  lift -> fail (False). The True->False flip localises the Point of No Return.

  Primary integrity is asserted LIVE in this run: around EVERY reset_to(env 1)
  the primary's cube pose is captured before/after; the max delta must be ~0,
  demonstrating the probe write never touches env 0.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_pnor_grasped.py --headless
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD dual-env grasped-region PoNR")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--num_probe_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=160)
parser.add_argument("--drop_window", type=int, default=22,
                    help="Steps to force the primary gripper OPEN once lifted.")
parser.add_argument("--drop_lift", type=float, default=0.10,
                    help="Cube must be this far above rest before the slip is injected [m].")
parser.add_argument("--probe_stride", type=int, default=10)
parser.add_argument("--probe_budget", type=int, default=140)
parser.add_argument("--lift_thresh", type=float, default=0.04)
parser.add_argument("--locality_tol", type=float, default=1e-6,
                    help="Max acceptable env0 pose delta across a probe reset_to [m].")
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
from ipfd.oracles.pick_lift_sm import PickAndLiftSm, sm_action  # noqa: E402
from ipfd import build_report  # noqa: E402
from ipfd.adapters.isaac_lab import offset_root_positions, slice_state  # noqa: E402
from ipfd.types import Rollout  # noqa: E402
from ipfd.ponr import point_of_no_return  # noqa: E402

PRIMARY_ENV = 0  # NEVER reset_to or action-corrupted by the probe
PROBE_ENV = 1    # receives exported snapshots; diverges freely
GRIPPER_CH = 7   # action = [pos(3), quat(4), gripper(1)]; OPEN=+1, CLOSE=-1
LIFT_STATE = 4   # PickSmState.LIFT_OBJECT (see ipfd.oracles.pick_lift_sm)


def log(msg: str) -> None:
    print(f"[grasped] {msg}", flush=True)


def n_identity(n: int, dev) -> torch.Tensor:
    a = torch.zeros((n, 8), dtype=torch.float32, device=dev)
    a[:, 3] = 1.0
    return a


def obj_pose(env, i: int) -> torch.Tensor:
    return wp.to_torch(env.unwrapped.scene["object"].data.root_pose_w)[i].clone()


def obj_z(env, i: int) -> float:
    return float(wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)[i, 2].item())


def probe_action(env, rsm, des_or) -> torch.Tensor:
    """Recovery SM drives the probe env; the primary is held at identity."""
    a = sm_action(env, rsm, des_or).clone()
    a[PRIMARY_ENV] = 0.0
    a[PRIMARY_ENV, 3] = 1.0
    return a


def record_primary(env, dt, dev, n, seed):
    """Drive the primary pick-lift in env 0, inject a GRASPED-REGION gripper slip,
    and record the rollout + per-step checkpoints. env 0 is never reset_to."""
    des_or = torch.zeros((n, 4), device=dev)
    des_or[:, 1] = 1.0
    env.reset(seed=seed)
    env.step(n_identity(n, dev))
    sm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)

    obs_list, act_list, objz, states, sm_snaps = [], [], [], [], []
    action = n_identity(n, dev)
    running_min = obj_z(env, PRIMARY_ENV)
    drop_step, drop_end = None, None
    for step in range(args.max_steps):
        z_now = obj_z(env, PRIMARY_ENV)
        running_min = min(running_min, z_now)
        # Inject the slip: first time the cube is genuinely grasped AND lifted,
        # force the primary gripper OPEN for drop_window steps -> cube falls.
        if (drop_step is None and int(sm.sm_state[PRIMARY_ENV].item()) == LIFT_STATE
                and z_now > running_min + args.drop_lift):
            drop_step, drop_end = step, step + args.drop_window
        if drop_step is not None and step < drop_end:
            action = action.clone()
            action[PRIMARY_ENV, GRIPPER_CH] = 1.0  # OPEN -> release the cube

        obs, _r, _term, _trunc, _i = env.step(action)
        obs_list.append(obs["policy"][PRIMARY_ENV].detach().cpu().numpy().reshape(-1).astype(np.float64))
        act_list.append(action[PRIMARY_ENV].detach().cpu().numpy().reshape(-1).astype(np.float64))
        objz.append(obj_z(env, PRIMARY_ENV))
        states.append(slice_state(env.unwrapped.scene.get_state(), slice(0, 1)))
        # SM progress that will drive the primary FROM this checkpoint. The
        # recovery oracle CONTINUES the policy (restored SM), a fair "can it still
        # finish from here" probe rather than restarting from REST.
        sm_snaps.append((int(sm.sm_state[PRIMARY_ENV].item()), float(sm.sm_wait_time[PRIMARY_ENV].item())))
        action = sm_action(env, sm, des_or)
    z0 = float(np.min(objz[: min(15, len(objz))]))
    return obs_list, act_list, objz, states, sm_snaps, z0, des_or, drop_step


def probe_checkpoint(env, state_env0, sm_snap, des_or, z0, dt, dev, n, loc):
    """Export env 0's checkpoint INTO the probe env and CONTINUE the primary
    policy there. True if the probe cell completes the lift. Also asserts, live,
    that the reset_to write does not perturb env 0 (loc accumulates max delta)."""
    origins = env.unwrapped.scene.env_origins
    delta = (origins[PROBE_ENV] - origins[PRIMARY_ENV]).detach()
    state_probe = offset_root_positions(state_env0, delta)

    pose0_before = obj_pose(env, PRIMARY_ENV)
    env.unwrapped.scene.reset_to(state_probe, torch.tensor([PROBE_ENV], device=dev, dtype=torch.long))
    pose0_after = obj_pose(env, PRIMARY_ENV)
    loc["max"] = max(loc["max"], float((pose0_after - pose0_before).abs().max().item()))
    loc["n"] += 1

    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0  # avoid timeout auto-reset mid-probe
    rsm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)
    rsm.sm_state[PROBE_ENV] = sm_snap[0]          # continue the policy, do not restart
    rsm.sm_wait_time[PROBE_ENV] = sm_snap[1]
    action = probe_action(env, rsm, des_or)
    for _ in range(args.probe_budget):
        env.step(action)
        if obj_z(env, PROBE_ENV) > z0 + args.lift_thresh:
            return True
        action = probe_action(env, rsm, des_or)
    return False


def evaluate_recovery(env, states, sm_snaps, des_or, z0, dt, dev, n, drop_step, loc):
    T = len(states)
    probe_steps = set(range(0, T, args.probe_stride))
    probe_steps.add(T - 1)
    if drop_step is not None:
        probe_steps.update(t for t in range(max(0, drop_step - 8), min(T, drop_step + 30)))
    probe_steps = sorted(probe_steps)
    verdict = {t: probe_checkpoint(env, states[t], sm_snaps[t], des_or, z0, dt, dev, n, loc)
               for t in probe_steps}
    rec = np.zeros(T, dtype=bool)
    last = True
    for t in range(T):
        if t in verdict:
            last = verdict[t]
        rec[t] = last
    return rec, probe_steps, verdict


def main() -> None:
    env = None
    st = {
        "second_environment_supported": "NO",
        "snapshot_transfer_supported": "NO",
        "primary_rollout_corruption": "UNKNOWN",
        "recovery_oracle_non_degenerate": "NO",
        "pnor_measurement_possible": "NO",
        "overall_status": "BLOCKED",
    }
    loc = {"max": 0.0, "n": 0}
    try:
        n = 1 + max(1, args.num_probe_envs)
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=n)
        env = gym.make(args.env_id, cfg=env_cfg)
        dev = env.unwrapped.device
        dt = env_cfg.sim.dt * env_cfg.decimation
        st["second_environment_supported"] = "YES"  # vectorised pool, env1 stepped independently
        log(f"env={args.env_id} num_envs={n} (env0=primary pristine, env1=probe) dt={dt:.4f}")

        (obs_list, act_list, objz, states, sm_snaps, z0,
         des_or, drop_step) = record_primary(env, dt, dev, n, seed=0)
        T = len(states)
        zmax = float(np.max(objz))
        z_end = objz[-1]
        primary_failed = z_end < z0 + args.lift_thresh
        log(f"primary: T={T} z0={z0:.3f} zmax={zmax:.3f} z_end={z_end:.3f} "
            f"drop_step={drop_step} primary_failed={primary_failed}")
        if drop_step is None:
            log("WARN: gripper slip was never injected (cube never reached LIFT+drop_lift).")

        rec, probe_steps, verdict = evaluate_recovery(
            env, states, sm_snaps, des_or, z0, dt, dev, n, drop_step, loc)
        st["snapshot_transfer_supported"] = "YES"  # get_state -> offset -> reset_to executed

        n_true, n_false = int(rec.sum()), int((~rec).sum())
        ponr = point_of_no_return(rec)

        rollout = Rollout(
            observations=np.asarray(obs_list), actions=np.asarray(act_list),
            entropy=None, embeddings=None, success=not primary_failed,
            t_failure=None if not primary_failed else T - 1,
            recovery_success=rec, dt=dt, seed=0,
            meta={"source": "isaac_lab", "robot": "franka", "task": "lift",
                  "policy": "scripted_pick_lift", "failure": "grasped_gripper_slip",
                  "drop_step": drop_step, "dual_env_probe": True},
        )
        report = build_report(rollout)

        log("raw probe verdicts: " + ", ".join(f"{t}:{'T' if verdict[t] else 'F'}" for t in probe_steps))
        log(f"recover(True/False) = {n_true}/{n_false}  PoNR={ponr}  drop_step={drop_step}  "
            f"alarm={report.t_alarm}")
        log(f"primary-integrity: reset_to x{loc['n']} into env1, max env0 pose delta = "
            f"{loc['max']:.2e} m (tol {args.locality_tol})")

        # --- verdicts ---
        primary_pristine = loc["max"] <= args.locality_tol
        st["primary_rollout_corruption"] = "NO" if primary_pristine else "YES"
        st["recovery_oracle_non_degenerate"] = "YES" if (n_true > 0 and n_false > 0) else "NO"

        ponr_ok = (
            drop_step is not None and ponr is not None and ponr > 0
            and abs(ponr - drop_step) <= args.probe_stride + args.drop_window
        )
        st["pnor_measurement_possible"] = "YES" if ponr_ok else "NO"
        if ponr is not None and drop_step is not None and rollout.t_failure is not None:
            lead = (rollout.t_failure - ponr) * dt
            log(f"PoNR {ponr} vs injected slip {drop_step} (tol {args.probe_stride + args.drop_window}); "
                f"PoNR lead over observable failure = {lead:+.2f}s")

        all_yes = (
            st["second_environment_supported"] == "YES"
            and st["snapshot_transfer_supported"] == "YES"
            and st["primary_rollout_corruption"] == "NO"
            and st["recovery_oracle_non_degenerate"] == "YES"
            and st["pnor_measurement_possible"] == "YES"
        )
        st["overall_status"] = "VERIFIED" if all_yes else "PARTIAL"
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
        print("DUAL_PROBE_STATUS:")
        for k in ("second_environment_supported", "snapshot_transfer_supported",
                  "primary_rollout_corruption", "recovery_oracle_non_degenerate",
                  "pnor_measurement_possible", "overall_status"):
            print(f"- {k}: {st[k]}")
        print("=" * 55, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
