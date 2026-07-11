"""Tasks 2+3: drive IPFD with a REAL competent policy on real Isaac Lab rollouts.

No trained checkpoint exists on this machine, so the strongest available policy
(priority: trained -> scripted -> heuristic) is Isaac Lab's own scripted
pick-and-lift state machine, imported from :mod:`ipfd.oracles.pick_lift_sm`
(BSD-3-Clause, Isaac Lab, reproduced verbatim) so the controller is exactly the
proven reference, not a reimplementation that might silently mis-grasp.

Experiment (env ``Isaac-Lift-Cube-Franka-IK-Abs-v0``, single env, real physics):

  * NOMINAL rollout: the scripted policy lifts the cube -> SUCCESS. IPFD should
    report NO point of no return (nothing was ever irrecoverable) and raise no
    false failure alarm. This is the negative control.

  * PERTURBED rollout(s): at a KNOWN step ``t_perturb`` we physically displace
    the cube out of the arm's reach (a real write to the sim, ground-truth known).
    The recovery oracle is a FRESH scripted pick-lift restarted from the saved
    sim state: it succeeds while the cube is reachable and fails once it is not,
    so ``recovery_success`` flips True->False around ``t_perturb`` and IPFD's PoNR
    should land there -- NOT at step 0.

This validates IPFD's recovery-probe PoNR against ground truth on real physics.
It is a debugger-validation experiment: the failure is INJECTED and its timing is
known, which is exactly how you check whether the detector finds the right moment.
The scripted policy exposes no confidence/latent signal, so only the
action-variance detector and the recovery-probe PoNR are exercised here.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_real_policy.py --headless
    # add --diagnose to run the scripted policy uninterrupted (sanity check)

RESULT (2026-07-04, Isaac Lab 4.5.22, an NVIDIA Blackwell-class consumer GPU) -- honest, not the hoped-for one:

  * ``--diagnose`` (scripted policy, NO probes): lifts the cube on every seed
    (+0.23..+0.41 m, reaches all SM states 0->4). The controller is competent.

  * Full run (recovery probe interleaved in the SAME env): the nominal rollout
    FAILS to lift (z_end < z0) and PoNR is degenerate/spurious. Interleaving the
    contact-rich recovery probe (save state -> grasp+lift with a fresh SM ->
    reset_to) reproducibly corrupts the primary rollout. THREE different probe
    implementations (buffer-order fix, non-yanking first action, larger budget)
    produced byte-identical degenerate output, so the failure is deterministic
    and dominated by something the probe parameters do not control -- most likely
    contact/solver state that ``get_state``/``reset_to`` do not capture after a
    grasp (single-step restore is nonetheless bit-exact; see
    ``verify_state_fidelity.py``).

  Conclusion: with a real policy, IPFD's analysis + single-step state restore are
  sound, but the recovery-probe PoNR does NOT yet yield a meaningful point of no
  return in the loop. meaningful_pnor_detected = NO. This is an open limitation of
  the recovery-probe design, not of the detectors or the analysis layer.
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD real-policy validation")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
parser.add_argument("--max_steps", type=int, default=170)
parser.add_argument("--perturb_step", type=int, default=55)
parser.add_argument("--probe_stride", type=int, default=25)
parser.add_argument("--probe_budget", type=int, default=200, help="Steps a fresh SM gets to recover.")
parser.add_argument("--lift_thresh", type=float, default=0.04, help="Object rise [m] counted as a lift.")
parser.add_argument("--reach_push", type=float, default=1.2, help="Out-of-reach displacement [m] in +x.")
parser.add_argument("--diagnose", action="store_true",
                    help="Run the scripted policy UNINTERRUPTED (no probes/perturb) and report max lift per seed.")
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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from ipfd import build_report  # noqa: E402
from ipfd.types import Rollout  # noqa: E402
from ipfd.ponr import point_of_no_return  # noqa: E402

wp.init()


from ipfd.oracles.pick_lift_sm import (  # noqa: E402
    PickAndLiftSm,
    sm_action,
    object_z,
    identity_action as _identity_action,
)


def log(msg: str) -> None:
    print(f"[real-policy] {msg}", flush=True)


def displace_object_out_of_reach(env, push: float) -> None:
    """Real sim write: teleport the cube +x by `push` metres (ground-truth doom)."""
    obj = env.unwrapped.scene["object"]
    pose = wp.to_torch(obj.data.root_pose_w).clone()
    pose[:, 0] += push
    obj.write_root_pose_to_sim(pose)
    vel = wp.to_torch(obj.data.root_vel_w).clone() * 0.0
    obj.write_root_velocity_to_sim(vel)


def probe_recovery(env, desired_orientation, z0, dt, dev) -> bool:
    """Save state, run a FRESH scripted pick-lift for the budget, restore. True if it lifts."""
    S = env.unwrapped.scene.get_state()
    ep = None
    if hasattr(env.unwrapped, "episode_length_buf"):
        ep = env.unwrapped.episode_length_buf.clone()
    sm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)
    # First action holds the CURRENT pose via the SM's REST state -- never the
    # identity pose (0,0,0), which for an IK-Abs controller yanks the arm to the
    # base origin, wastes the recovery budget, and its contact dynamics make the
    # subsequent reset_to() restore imperfect. Buffers are fresh here (called
    # right after a real env.step), so sm_action reads the true current pose.
    action = sm_action(env, sm, desired_orientation)
    recovered = False
    try:
        for _ in range(args.probe_budget):
            env.step(action)
            if object_z(env) > z0 + args.lift_thresh:
                recovered = True
                break
            action = sm_action(env, sm, desired_orientation)
    finally:
        env.unwrapped.scene.reset_to(S, None)
        if ep is not None:
            env.unwrapped.episode_length_buf[:] = ep
    return recovered


def run_rollout(env, dt, dev, seed, perturb_step=None):
    """One primary rollout with a strided recovery probe. Returns (Rollout, diag)."""
    desired_orientation = torch.zeros((1, 4), device=dev)
    desired_orientation[:, 1] = 1.0
    env.reset(seed=seed)
    # settle one step, capture resting object height
    env.step(_identity_action(dev))
    z0 = object_z(env)
    sm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)

    obs_list, act_list, rec_flags, objz = [], [], [], []
    action = _identity_action(dev)
    last_rec = True
    perturbed_done = False
    for step in range(args.max_steps):
        if perturb_step is not None and step == perturb_step and not perturbed_done:
            displace_object_out_of_reach(env, args.reach_push)
            perturbed_done = True

        obs, _rew, term, trunc, _info = env.step(action)
        obs_list.append(obs["policy"].detach().cpu().numpy().reshape(-1).astype(np.float64))
        act_list.append(action.detach().cpu().numpy().reshape(-1).astype(np.float64))
        objz.append(object_z(env))

        # Compute the next action from FRESH post-step buffers BEFORE any probe.
        # A probe calls reset_to(), which restores physical state but leaves the
        # .data sensor buffers stale until the next env.step -- reading them here
        # would feed the primary a wrong target and corrupt the rollout.
        next_action = sm_action(env, sm, desired_orientation)

        # Probe on the stride, plus a dense burst right after the injected doom
        # so the True->False flip is captured even if the episode terminates soon.
        do_probe = (step % args.probe_stride == 0)
        if perturb_step is not None and 0 <= step - perturb_step <= 9 and (step - perturb_step) % 3 == 0:
            do_probe = True
        if do_probe:
            last_rec = probe_recovery(env, desired_orientation, z0, dt, dev)
        rec_flags.append(last_rec)

        action = next_action
        if bool(torch.as_tensor(term).any()) or bool(torch.as_tensor(trunc).any()):
            break

    T = len(obs_list)
    lifted_end = objz[-1] > z0 + args.lift_thresh
    success = lifted_end and perturb_step is None
    recovery = np.asarray(rec_flags[:T], dtype=bool)
    rollout = Rollout(
        observations=np.asarray(obs_list),
        actions=np.asarray(act_list),
        entropy=None,
        embeddings=None,
        success=success,
        t_failure=None if success else T - 1,
        recovery_success=recovery,
        dt=dt,
        seed=seed,
        meta={"source": "isaac_lab", "robot": "franka", "task": "lift",
              "policy": "scripted_pick_lift", "perturb_step": perturb_step},
    )
    diag = {"T": T, "z0": z0, "z_end": objz[-1], "lifted_end": lifted_end,
            "n_recover_true": int(recovery.sum()), "n_recover_false": int((~recovery).sum())}
    return rollout, diag


def main() -> None:
    env = None
    status = {
        "state_restore_fidelity": "SEE_verify_state_fidelity",  # measured by the other script
        "meaningful_pnor_detected": "NO",
        "recovery_oracle_non_degenerate": "NO",
        "measurable_failure_lead_time": "NO",
        "overall_status": "PARTIALLY_VERIFIED",
    }
    try:
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=1)
        env = gym.make(args.env_id, cfg=env_cfg)
        dev = env.unwrapped.device
        dt = env_cfg.sim.dt * env_cfg.decimation
        log(f"env={args.env_id} dt={dt:.4f} act_dim={env.action_space.shape[-1]}")

        if args.diagnose:
            desired_orientation = torch.zeros((1, 4), device=dev)
            desired_orientation[:, 1] = 1.0
            for seed in range(4):
                env.reset(seed=seed)
                env.step(_identity_action(dev))
                z0 = object_z(env)
                sm = PickAndLiftSm(dt, 1, dev, position_threshold=0.01)
                action = _identity_action(dev)
                zmax = z0
                states = []
                for _step in range(args.max_steps):
                    env.step(action)
                    zmax = max(zmax, object_z(env))
                    states.append(int(sm.sm_state[0].item()))
                    action = sm_action(env, sm, desired_orientation)
                log(f"diagnose seed={seed}: z0={z0:.3f} zmax={zmax:.3f} lift={zmax - z0:+.3f} "
                    f"final_sm_state={states[-1]} reached_states={sorted(set(states))}")
            env.close()
            env = None
            print("\nDIAGNOSE_DONE", flush=True)
            return

        rollouts = [
            ("nominal", run_rollout(env, dt, dev, seed=0, perturb_step=None)),
            ("perturbed@55", run_rollout(env, dt, dev, seed=1, perturb_step=args.perturb_step)),
            ("perturbed@75", run_rollout(env, dt, dev, seed=2, perturb_step=75)),
        ]

        any_meaningful_ponr = False
        any_recover_true = any_recover_false = False
        any_lead_time = False
        for name, (rollout, diag) in rollouts:
            report = build_report(rollout)
            ponr = point_of_no_return(rollout.recovery_success)
            log(f"--- {name}: T={diag['T']} z0={diag['z0']:.3f} z_end={diag['z_end']:.3f} "
                f"lifted_end={diag['lifted_end']} success={rollout.success} "
                f"recover(T/F)={diag['n_recover_true']}/{diag['n_recover_false']}")
            log(f"    PoNR={report.t_ponr} alarm={report.t_alarm} t_failure={rollout.t_failure} "
                f"perturb={rollout.meta['perturb_step']}")
            for line in report.summary().splitlines():
                log("    " + line)

            any_recover_true |= diag["n_recover_true"] > 0
            any_recover_false |= diag["n_recover_false"] > 0
            if rollout.meta["perturb_step"] is not None and ponr is not None and ponr > 0:
                any_meaningful_ponr = True
                # is PoNR near the injected doom? (within one probe stride)
                near = abs(ponr - rollout.meta["perturb_step"]) <= args.probe_stride
                log(f"    -> PoNR {ponr} vs injected doom {rollout.meta['perturb_step']} "
                    f"(near={near}, tol={args.probe_stride})")
            if (report.t_alarm is not None and rollout.t_failure is not None
                    and rollout.t_failure > report.t_alarm):
                any_lead_time = True
                lead = (rollout.t_failure - report.t_alarm) * dt
                log(f"    -> failure lead time = +{lead:.2f}s (alarm before observable failure)")

        status["meaningful_pnor_detected"] = "YES" if any_meaningful_ponr else "NO"
        status["recovery_oracle_non_degenerate"] = "YES" if (any_recover_true and any_recover_false) else "NO"
        status["measurable_failure_lead_time"] = "YES" if any_lead_time else "NO"
        core = (any_meaningful_ponr and any_recover_true and any_recover_false)
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
        print("IPFD_REAL_POLICY_STATUS:")
        print(f"- state_restore_fidelity: {status['state_restore_fidelity']}")
        print(f"- meaningful_pnor_detected: {status['meaningful_pnor_detected']}")
        print(f"- recovery_oracle_non_degenerate: {status['recovery_oracle_non_degenerate']}")
        print(f"- measurable_failure_lead_time: {status['measurable_failure_lead_time']}")
        print(f"- overall_status: {status['overall_status']}")
        print("=" * 55, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
