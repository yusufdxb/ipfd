"""Task 1: measure state save/restore fidelity on a live Isaac Lab env.

IPFD's Point of No Return rests entirely on the recovery probe, which itself
rests on ``scene.get_state()`` / ``scene.reset_to()`` faithfully round-tripping
the simulator. If restore is lossy, every PoNR is biased. So we MEASURE it, we
do not assume it.

Two independent checks, both on the live ``Isaac-Lift-Cube-Franka-v0`` sim:

  A. Write-back fidelity: ``S = get_state()``; ``reset_to(S)``; ``get_state()``
     again; the two state dicts must be identical (did restore write what we
     saved?).

  B. Replay determinism: from a fixed state, apply a fixed action a* and record
     (obs1, reward1, term1, next-state P1). Restore, apply the SAME a*, record
     (obs2, reward2, term2, P2). If restore is faithful and stepping is
     deterministic, the two replays must match. This is exactly the property
     the recovery probe depends on.

Prints a machine-readable STATE_RESTORE_FIDELITY block. Tolerances are reported
alongside the measured max-abs-diff so the numbers speak for themselves.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
    /path/to/isaac-lab/python scripts/verify_state_fidelity.py --headless
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* import, so isort reordering is disabled here.

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IPFD state save/restore fidelity")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--warmup", type=int, default=25, help="Steps to reach a non-trivial state.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tol", type=float, default=1e-4, help="Max-abs-diff pass tolerance.")
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

RESULTS = {
    "observations_match": "FAIL",
    "joint_states_match": "FAIL",
    "object_states_match": "FAIL",
    "rewards_match": "FAIL",
    "overall": "FAIL",
}
NOTES: list[str] = []


def log(msg: str) -> None:
    print(f"[fidelity] {msg}", flush=True)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max().item())


def snapshot_entities(env) -> dict[str, torch.Tensor]:
    """Robot joint state + object root state, read from live .data buffers."""
    scene = env.unwrapped.scene
    robot = scene["robot"].data
    obj = scene["object"].data
    # .data fields are Warp arrays in Isaac Lab 4.5.x -> wp.to_torch to read.
    return {
        "joint_pos": wp.to_torch(robot.joint_pos).clone(),
        "joint_vel": wp.to_torch(robot.joint_vel).clone(),
        "obj_pos": wp.to_torch(obj.root_pos_w).clone(),
        "obj_quat": wp.to_torch(obj.root_quat_w).clone(),
        "obj_vel": wp.to_torch(obj.root_vel_w).clone(),
    }


def state_maxdiff(s1: dict, s2: dict) -> dict[str, float]:
    out = {}
    for k in s1["articulation"]:
        for f in ("root_pose", "root_velocity", "joint_position", "joint_velocity"):
            out[f"art.{k}.{f}"] = maxdiff(s1["articulation"][k][f], s2["articulation"][k][f])
    for k in s1.get("rigid_object", {}):
        for f in ("root_pose", "root_velocity"):
            out[f"rigid.{k}.{f}"] = maxdiff(s1["rigid_object"][k][f], s2["rigid_object"][k][f])
    return out


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=1)
        env = gym.make(args.env_id, cfg=env_cfg)
        act_dim = int(env.action_space.shape[-1])
        dev = env.unwrapped.device

        env.reset(seed=args.seed)
        # Fixed, deterministic action sequences (no RNG at replay time).
        rng = np.random.default_rng(args.seed)
        warm = [torch.as_tensor(0.15 * rng.standard_normal((1, act_dim)), dtype=torch.float32, device=dev)
                for _ in range(args.warmup)]
        for a in warm:
            env.step(a)
        a_star = torch.as_tensor(0.2 * rng.standard_normal((1, act_dim)), dtype=torch.float32, device=dev)

        # --- Check A: write-back fidelity ---------------------------------
        S = env.unwrapped.scene.get_state()
        pre_entities = snapshot_entities(env)
        env.unwrapped.scene.reset_to(S, None)
        S_after = env.unwrapped.scene.get_state()
        wb = state_maxdiff(S, S_after)
        wb_max = max(wb.values())
        log(f"A. write-back max-abs-diff = {wb_max:.3e}  (per-field: "
            + ", ".join(f"{k}={v:.1e}" for k, v in wb.items()) + ")")

        # --- Restore to the SAVED state, then first replay ----------------
        env.unwrapped.scene.reset_to(S, None)
        obs1, rew1, term1, trunc1, info1 = env.step(a_star)
        P1 = snapshot_entities(env)
        o1 = obs1["policy"].clone()
        r1 = float(torch.as_tensor(rew1).reshape(-1)[0].item())

        # --- Restore AGAIN to the same saved state, then identical replay -
        env.unwrapped.scene.reset_to(S, None)
        obs2, rew2, term2, trunc2, info2 = env.step(a_star)
        P2 = snapshot_entities(env)
        o2 = obs2["policy"].clone()
        r2 = float(torch.as_tensor(rew2).reshape(-1)[0].item())

        # --- Compare ------------------------------------------------------
        obs_d = maxdiff(o1, o2)
        joint_d = max(maxdiff(P1["joint_pos"], P2["joint_pos"]), maxdiff(P1["joint_vel"], P2["joint_vel"]))
        obj_d = max(maxdiff(P1["obj_pos"], P2["obj_pos"]),
                    maxdiff(P1["obj_quat"], P2["obj_quat"]),
                    maxdiff(P1["obj_vel"], P2["obj_vel"]))
        rew_d = abs(r1 - r2)
        term_match = (bool(torch.as_tensor(term1).any()) == bool(torch.as_tensor(term2).any())
                      and bool(torch.as_tensor(trunc1).any()) == bool(torch.as_tensor(trunc2).any()))

        log(f"B. replay obs   max-abs-diff = {obs_d:.3e}")
        log(f"B. replay joint max-abs-diff = {joint_d:.3e}")
        log(f"B. replay obj   max-abs-diff = {obj_d:.3e}")
        log(f"B. replay reward diff        = {rew_d:.3e}  (r1={r1:.6f}, r2={r2:.6f})")
        log(f"B. termination flags match   = {term_match}  "
            f"(term {bool(torch.as_tensor(term1).any())}/{bool(torch.as_tensor(term2).any())}, "
            f"trunc {bool(torch.as_tensor(trunc1).any())}/{bool(torch.as_tensor(trunc2).any())})")
        # sanity: warmup actually produced motion (non-static object/joints)
        moved = maxdiff(pre_entities["joint_pos"], S["articulation"]["robot"]["joint_position"])
        log(f"sanity: |joint_pos(pre) - state.joint_position| = {moved:.3e} (buffers agree ~0 expected)")

        tol = args.tol
        RESULTS["observations_match"] = "PASS" if obs_d <= tol else "FAIL"
        RESULTS["joint_states_match"] = "PASS" if joint_d <= tol else "FAIL"
        RESULTS["object_states_match"] = "PASS" if obj_d <= tol else "FAIL"
        RESULTS["rewards_match"] = "PASS" if rew_d <= tol else "FAIL"
        write_back_ok = wb_max <= tol
        if not write_back_ok:
            NOTES.append(f"write-back diff {wb_max:.3e} exceeds tol {tol:.0e}: reset_to did not "
                         "reproduce the saved state exactly.")
        RESULTS["overall"] = "PASS" if (write_back_ok and term_match and all(
            RESULTS[k] == "PASS" for k in
            ("observations_match", "joint_states_match", "object_states_match", "rewards_match"))) else "FAIL"

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
        print("\n" + "=" * 50)
        if NOTES:
            print("NOTES:")
            for n in NOTES:
                print(f"  - {n}")
            print("-" * 50)
        print("STATE_RESTORE_FIDELITY:")
        for k in ("observations_match", "joint_states_match", "object_states_match", "rewards_match", "overall"):
            print(f"- {k}: {RESULTS[k]}")
        print("=" * 50, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
