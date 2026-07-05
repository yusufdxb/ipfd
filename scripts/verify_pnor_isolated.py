"""Meaningful PoNR under a real policy via ENV-ISOLATED recovery probing.

Chain of evidence:
  * verify_state_fidelity.py     -> single-step reset_to is bit-exact.
  * verify_probe_transparency.py -> in-loop probing corrupts the primary after a
                                    grasp (PhysX contact state not restored).
  * verify_pnor_decoupled.py     -> a single reset_to poisons a num_envs=1 sim
                                    even across env.reset() (reset_to_poisons_env).
  * verify_multienv_isolation.py -> BUT per-env reset_to is LOCAL: resetting env 1
                                    leaves env 0 bit-identical. => isolate probes.

This harness uses that isolation. With num_envs = 1 + P:
  PASS 1 (record): drive the primary pick-lift and record env 0's rollout +
    per-step checkpoints. env 0 is NEVER reset_to, so its rollout is pristine.
    The cube is displaced out of reach at a KNOWN step for the perturbed case.
  PASS 2 (probe): for a strided set of steps t, reset_to env 0's checkpoint S_t
    INTO a probe env, run a fresh pick-lift there for a budget, and read that
    probe env's lift. recovery_success[t] = did the probe env recover? env 0 is
    untouched throughout.

If reachable checkpoints recover (True) and out-of-reach ones do not (False), the
True->False flip localises PoNR at the injected doom -- NOT at step 0.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_pnor_isolated.py --headless
    # add --debug_step T to trace a single nominal probe verbosely

RESULT (2026-07-05, Isaac Lab 4.5.22, RTX 5070) -- major progress, honest partial:

  Env isolation SOLVES the corruption/poison problem that blocked every earlier
  attempt:
    * The primary rollout in env 0 is now PRISTINE and lifts (nominal zmax 0.341,
      success=True) -- it is never reset_to, so it is never poisoned.
    * Per-env reset_to is local (verify_multienv_isolation.py): env 0 stays
      bit-identical while env 1 is churned through reset_to.
    * get_state() poses are ABSOLUTE, so a checkpoint must be origin-shifted into
      the probe env's cell (else the two arms collide).
    * The recovery oracle CONTINUES the primary policy (restored SM progress)
      rather than restarting from REST (which would drop a held cube). With this,
      GRASPED-region recovery verdicts are correct (post-grasp checkpoints -> lift
      -> True).

  Remaining honest gap: meaningful_pnor_detected = NO. Early, PRE-GRASP
  checkpoints stall in APPROACH_OBJECT: --debug_step 30 shows the probe SM reaches
  state 2 and never grasps (ee never gets within the 1 cm threshold; it nudges the
  cube instead). Cause: reset_to hands the probe a COLD PhysX contact/solver state
  (the same limitation measured in verify_probe_transparency.py), which is enough
  to derail the scripted grasp's sub-centimetre approach. So pre-grasp recovery
  verdicts are unreliable and the True->False sequence is not clean enough for
  point_of_no_return to fire.

  Bottom line: IPFD's analysis layer, single-step restore, and env-isolated
  probing are all sound; the residual blocker is recovery-oracle robustness to a
  cold-contact restart during fine manipulation -- a controller/policy property,
  not an IPFD infrastructure gap. A concrete next step (untried here, needs
  go-ahead): inject the failure in the GRASPED region (e.g. drop the cube after
  lift) where the continue-oracle is already reliable, to get a clean PoNR flip.
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD env-isolated PoNR")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--num_probe_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=150)
parser.add_argument("--perturb_step", type=int, default=60)
parser.add_argument("--probe_stride", type=int, default=10)
parser.add_argument("--probe_budget", type=int, default=140)
parser.add_argument("--lift_thresh", type=float, default=0.04)
parser.add_argument("--reach_push", type=float, default=1.2)
parser.add_argument("--debug_step", type=int, default=-1,
                    help="If >=0: probe only this nominal checkpoint verbosely, then exit.")
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
from ipfd import build_report  # noqa: E402
from ipfd.types import Rollout  # noqa: E402
from ipfd.ponr import point_of_no_return  # noqa: E402

PROBE_ENV = 1  # env 0 is the untouched primary; probes run in env 1


def log(msg: str) -> None:
    print(f"[isolated] {msg}", flush=True)


def n_identity(n: int, dev) -> torch.Tensor:
    a = torch.zeros((n, 8), dtype=torch.float32, device=dev)
    a[:, 3] = 1.0
    return a


def obj_z(env, i: int) -> float:
    return float(wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)[i, 2].item())


def slice_state(state, idx):
    if torch.is_tensor(state):
        return state[idx].clone()
    if isinstance(state, dict):
        return {k: slice_state(v, idx) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(slice_state(v, idx) for v in state)
    return state


def offset_root_positions(state, delta):
    """get_state() poses are ABSOLUTE world coords. To drop env 0's checkpoint into
    the probe env's own cell (not on top of env 0), shift every root position by
    delta = origin(probe) - origin(0). Joints/velocities are origin-independent."""
    s = slice_state(state, slice(None))  # deep clone
    for grp in ("articulation", "rigid_object"):
        for _name, fields in s.get(grp, {}).items():
            if isinstance(fields, dict) and "root_pose" in fields:
                fields["root_pose"][:, :3] += delta
    return s


def displace_env0(env, push: float) -> None:
    """Teleport ONLY env 0's cube +x by push metres (ground-truth doom)."""
    obj = env.unwrapped.scene["object"]
    pose = wp.to_torch(obj.data.root_pose_w).clone()
    pose[0, 0] += push
    obj.write_root_pose_to_sim(pose)
    vel = wp.to_torch(obj.data.root_vel_w).clone()
    vel[0] = 0.0
    obj.write_root_velocity_to_sim(vel)


def probe_action(env, rsm, des_or, dev) -> torch.Tensor:
    """Recovery SM drives the probe env; env 0 is held at identity (ignored)."""
    a = sm_action(env, rsm, des_or).clone()
    a[0] = 0.0
    a[0, 3] = 1.0
    return a


def record_primary(env, dt, dev, n, seed, perturb_step):
    des_or = torch.zeros((n, 4), device=dev)
    des_or[:, 1] = 1.0
    env.reset(seed=seed)
    env.step(n_identity(n, dev))
    sm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)

    obs_list, act_list, objz, states, sm_snaps = [], [], [], [], []
    action = n_identity(n, dev)
    for step in range(args.max_steps):
        if perturb_step is not None and step == perturb_step:
            displace_env0(env, args.reach_push)
        obs, _r, _term, _trunc, _i = env.step(action)
        obs_list.append(obs["policy"][0].detach().cpu().numpy().reshape(-1).astype(np.float64))
        act_list.append(action[0].detach().cpu().numpy().reshape(-1).astype(np.float64))
        objz.append(obj_z(env, 0))
        states.append(slice_state(env.unwrapped.scene.get_state(), slice(0, 1)))  # env 0, count 1
        action = sm_action(env, sm, des_or)  # advances the primary SM to its next action
        # SM progress that will drive the primary FROM this checkpoint (the recovery
        # oracle CONTINUES the policy, rather than restarting from REST and dropping
        # any held cube -- a fair "can the policy still finish from here" probe).
        sm_snaps.append((int(sm.sm_state[0].item()), float(sm.sm_wait_time[0].item())))
    z0 = float(np.min(objz[: min(15, len(objz))]))
    return obs_list, act_list, objz, states, sm_snaps, z0, des_or


def probe_checkpoint(env, state_env0, sm_snap, des_or, z0, dt, dev, n) -> bool:
    """Reset env 0's checkpoint INTO the probe env, then CONTINUE the primary policy
    (restored SM progress) there. True if the probe env completes the lift."""
    origins = env.unwrapped.scene.env_origins
    delta = (origins[PROBE_ENV] - origins[0]).detach()
    state_probe = offset_root_positions(state_env0, delta)
    env.unwrapped.scene.reset_to(state_probe, torch.tensor([PROBE_ENV], device=dev, dtype=torch.long))
    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0  # avoid timeout auto-reset mid-probe
    rsm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)
    rsm.sm_state[PROBE_ENV] = sm_snap[0]          # continue the policy, do not restart
    rsm.sm_wait_time[PROBE_ENV] = sm_snap[1]
    action = probe_action(env, rsm, des_or, dev)
    for _ in range(args.probe_budget):
        env.step(action)
        if obj_z(env, PROBE_ENV) > z0 + args.lift_thresh:
            return True
        action = probe_action(env, rsm, des_or, dev)
    return False


def evaluate_recovery(env, states, sm_snaps, des_or, z0, dt, dev, n, perturb_step):
    T = len(states)
    probe_steps = set(range(0, T, args.probe_stride))
    probe_steps.add(T - 1)
    if perturb_step is not None:
        probe_steps.update(t for t in range(max(0, perturb_step - 5), min(T, perturb_step + 15)))
    probe_steps = sorted(probe_steps)
    verdict = {t: probe_checkpoint(env, states[t], sm_snaps[t], des_or, z0, dt, dev, n) for t in probe_steps}
    rec = np.zeros(T, dtype=bool)
    last = True
    for t in range(T):
        if t in verdict:
            last = verdict[t]
        rec[t] = last
    return rec, probe_steps, verdict


def run_case(env, name, dt, dev, n, seed, perturb_step, acc):
    obs_list, act_list, objz, states, sm_snaps, z0, des_or = record_primary(env, dt, dev, n, seed, perturb_step)
    T = len(states)
    rec, probe_steps, verdict = evaluate_recovery(env, states, sm_snaps, des_or, z0, dt, dev, n, perturb_step)

    zmax = float(np.max(objz))
    lifted_end = objz[-1] > z0 + args.lift_thresh
    success = lifted_end and perturb_step is None
    rollout = Rollout(
        observations=np.asarray(obs_list), actions=np.asarray(act_list),
        entropy=None, embeddings=None, success=success,
        t_failure=None if success else T - 1, recovery_success=rec, dt=dt, seed=seed,
        meta={"source": "isaac_lab", "robot": "franka", "task": "lift",
              "policy": "scripted_pick_lift", "perturb_step": perturb_step, "isolated": True},
    )
    report = build_report(rollout)
    ponr = point_of_no_return(rec)
    n_true, n_false = int(rec.sum()), int((~rec).sum())
    log(f"--- {name}: T={T} z0={z0:.3f} zmax={zmax:.3f} lifted_primary={zmax > z0 + args.lift_thresh} "
        f"success={success} recover(T/F)={n_true}/{n_false} probes={len(probe_steps)}")
    log("    raw probe verdicts: " + ", ".join(f"{t}:{'T' if verdict[t] else 'F'}" for t in probe_steps))
    log(f"    PoNR={ponr} injected_doom={perturb_step} alarm={report.t_alarm} t_failure={rollout.t_failure}")

    acc["any_true"] |= n_true > 0
    acc["any_false"] |= n_false > 0
    if perturb_step is not None and ponr is not None and ponr > 0:
        acc["meaningful_ponr"] = True
        near = abs(ponr - perturb_step) <= args.probe_stride
        acc["ponr_near_doom"] |= near
        lead = (rollout.t_failure - ponr) * dt if rollout.t_failure is not None else None
        log(f"    -> PoNR {ponr} vs injected doom {perturb_step} (near={near}, tol={args.probe_stride}); "
            f"PoNR lead over observable failure = {lead:+.2f}s" if lead is not None else "")


def main() -> None:
    env = None
    status = {"state_restore_fidelity": "PASS (verify_state_fidelity.py)",
              "per_env_isolation": "PASS (verify_multienv_isolation.py)",
              "meaningful_pnor_detected": "NO", "recovery_oracle_non_degenerate": "NO",
              "ponr_localizes_at_injected_doom": "NO", "overall_status": "PARTIALLY_VERIFIED"}
    acc = {"any_true": False, "any_false": False, "meaningful_ponr": False, "ponr_near_doom": False}
    try:
        n = 1 + max(1, args.num_probe_envs)
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=n)
        env = gym.make(args.env_id, cfg=env_cfg)
        dev = env.unwrapped.device
        dt = env_cfg.sim.dt * env_cfg.decimation
        log(f"env={args.env_id} num_envs={n} (env0=primary, env{PROBE_ENV}=probe) dt={dt:.4f}")

        if args.debug_step >= 0:
            t = args.debug_step
            obs_l, _a, objz, states, sm_snaps, z0, des_or = record_primary(env, dt, dev, n, 0, None)
            log(f"DEBUG probe checkpoint t={t}: primary sm_state@t={sm_snaps[t]}, "
                f"env0 objz@t={objz[t]:.3f}, z0={z0:.3f}, lift_target={z0 + args.lift_thresh:.3f}")
            origins = env.unwrapped.scene.env_origins
            delta = (origins[PROBE_ENV] - origins[0]).detach()
            env.unwrapped.scene.reset_to(offset_root_positions(states[t], delta),
                                         torch.tensor([PROBE_ENV], device=dev, dtype=torch.long))
            if hasattr(env.unwrapped, "episode_length_buf"):
                env.unwrapped.episode_length_buf[:] = 0
            rsm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)
            rsm.sm_state[PROBE_ENV] = sm_snaps[t][0]
            rsm.sm_wait_time[PROBE_ENV] = sm_snaps[t][1]
            action = probe_action(env, rsm, des_or, dev)
            for k in range(args.probe_budget):
                env.step(action)
                if k % 15 == 0 or obj_z(env, PROBE_ENV) > z0 + args.lift_thresh:
                    log(f"    k={k:3d} probe_sm_state={int(rsm.sm_state[PROBE_ENV].item())} "
                        f"objz(env1)={obj_z(env, PROBE_ENV):.3f}")
                if obj_z(env, PROBE_ENV) > z0 + args.lift_thresh:
                    log(f"    -> RECOVERED at k={k}")
                    break
                action = probe_action(env, rsm, des_or, dev)
            env.close(); env = None
            print("DEBUG_DONE", flush=True)
            return

        run_case(env, "nominal", dt, dev, n, seed=0, perturb_step=None, acc=acc)
        run_case(env, f"perturbed@{args.perturb_step}", dt, dev, n, seed=1,
                 perturb_step=args.perturb_step, acc=acc)

        status["meaningful_pnor_detected"] = "YES" if acc["meaningful_ponr"] else "NO"
        status["recovery_oracle_non_degenerate"] = "YES" if (acc["any_true"] and acc["any_false"]) else "NO"
        status["ponr_localizes_at_injected_doom"] = "YES" if acc["ponr_near_doom"] else "NO"
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
        print("IPFD_ISOLATED_PNOR_STATUS:")
        for k in ("state_restore_fidelity", "per_env_isolation", "meaningful_pnor_detected",
                  "recovery_oracle_non_degenerate", "ponr_localizes_at_injected_doom", "overall_status"):
            print(f"- {k}: {status[k]}")
        print("=" * 55, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
