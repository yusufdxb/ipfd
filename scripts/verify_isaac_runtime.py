"""Minimal Isaac Lab runtime smoke test for IPFD.

Scope is deliberately small. This script answers only:

    Does IPFD import against a live Isaac Lab install, and does its analysis layer
    attach to a REAL Isaac Lab rollout (observation structure, obs extraction,
    build_report) without mocks?

It does **not** exercise the recovery probe, the Point of No Return, or any
learned policy. For that -- the canonical end-to-end validation on a competent
trained policy, with the env-isolated recovery probe and measured PoNR -- see
``scripts/verify_learned_policy.py``.

  * ``verify_isaac_runtime.py``  -> runtime compatibility smoke test (this file).
  * ``verify_learned_policy.py`` -> canonical end-to-end validation on a competent
                                    learned policy (packaged ``collect_rollout``).

Run (real Isaac Lab install + GPU required):

    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_isaac_runtime.py --headless

The final block printed is machine-readable (IPFD_RUNTIME_SMOKE).
"""

from __future__ import annotations

# ruff: noqa: I001, E402
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* / isaaclab_tasks import, so isort reordering is disabled here.

import argparse
import os
import sys
import traceback

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    # Only handle the *absence* of Isaac Lab itself. A ModuleNotFoundError raised
    # from deeper inside a broken install (a missing sub-dependency) is a real
    # error and must surface, not be masked behind the friendly message.
    if exc.name != "isaaclab":
        raise
    print(
        "Isaac Lab was not found.\n\n"
        "This verification requires Isaac Lab 4.5.22.\n\n"
        "See the README installation section.",
        file=sys.stderr,
    )
    sys.exit(1)

parser = argparse.ArgumentParser(description="IPFD Isaac Lab runtime smoke test")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-v0",
                    help="Franka single-object manipulation env (closest to pick-and-place).")
parser.add_argument("--steps", type=int, default=16, help="Number of smoke steps to run.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # this is a validation run, never interactive

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  (registers the Isaac-* gym envs)
from isaaclab_tasks.utils import parse_env_cfg

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from ipfd import build_report
from ipfd.types import Rollout
from ipfd.adapters import isaac_lab as ipfd_adapter

RESULTS = {
    "isaac_lab_import": "PASS",  # reaching here means the import + AppLauncher worked
    "real_env_execution": "FAIL",
    "observation_structure": "FAIL",
    "adapter_attachment": "FAIL",
    "overall": "FAIL",
}


def log(section: str, msg: str) -> None:
    print(f"[{section}] {msg}", flush=True)


def main() -> None:
    env = None
    try:
        # === 1. Create a real env, reset, step ============================
        log("1", f"creating live env: {args.env_id} (num_envs=1, device={args.device})")
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=1)
        env = gym.make(args.env_id, cfg=env_cfg)
        real_module = type(env.unwrapped).__module__.startswith("isaaclab")
        log("1", f"env class: {type(env.unwrapped).__module__}.{type(env.unwrapped).__name__} (isaaclab={real_module})")

        obs, info = env.reset(seed=args.seed)
        obs_is_dict = isinstance(obs, dict)
        act_dim = int(env.action_space.shape[-1])
        log("1", f"reset() -> obs type={type(obs).__name__}, act_dim={act_dim}")

        reward_seen = False
        obs_records: list[np.ndarray] = []
        for _ in range(args.steps):
            a = torch.as_tensor(0.1 * np.random.randn(1, act_dim), dtype=torch.float32, device=args.device)
            step_out = env.step(a)
            assert len(step_out) == 5, f"env.step returned {len(step_out)}-tuple, expected 5"
            step_obs, rew, _term, _trunc, _inf = step_out
            reward_seen = reward_seen or (rew is not None)
            obs_records.append(ipfd_adapter._extract_obs(step_obs, "policy"))
        log("1", f"{args.steps}x step(): 5-tuple OK, reward_present={reward_seen}")
        if real_module and reward_seen:
            RESULTS["real_env_execution"] = "PASS"

        # === 2. Observation structure ====================================
        obs_key_ok = (not obs_is_dict) or ("policy" in obs)
        vec = ipfd_adapter._extract_obs(obs, "policy")
        extract_ok = isinstance(vec, np.ndarray) and vec.ndim == 1 and vec.dtype == np.float64
        log("2", f"obs is dict={obs_is_dict}, 'policy' group present={obs_key_ok}, "
                  f"_extract_obs -> shape={vec.shape} dtype={vec.dtype} ok={extract_ok}")
        if obs_key_ok and extract_ok:
            RESULTS["observation_structure"] = "PASS"

        # === 3. Adapter attachment: analysis layer on a REAL rollout ======
        # Build a Rollout from the live observations (no recovery probe) and run the
        # packaged analysis end to end. This proves the analysis layer attaches to
        # real Isaac Lab data; it makes no claim about detection or PoNR.
        observations = np.asarray(obs_records)
        actions = 0.1 * np.random.randn(observations.shape[0], act_dim)
        rollout = Rollout(
            observations=observations, actions=actions,
            success=False, t_failure=observations.shape[0] - 1,
            dt=float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0)), seed=args.seed,
            meta={"source": "isaac_lab", "note": "runtime_smoke"},
        )
        report = build_report(rollout)
        attached = report is not None and rollout.T == observations.shape[0] and rollout.T > 0
        log("3", f"build_report() on live rollout OK: T={rollout.T}, verdict line present={report is not None}")
        if attached:
            RESULTS["adapter_attachment"] = "PASS"

        RESULTS["overall"] = "PASS" if all(
            RESULTS[k] == "PASS" for k in
            ("real_env_execution", "observation_structure", "adapter_attachment")
        ) else "FAIL"

    except Exception:
        log("ERROR", "smoke test aborted with exception:")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        print("\n" + "=" * 60)
        print("IPFD_RUNTIME_SMOKE:")
        for k in ("isaac_lab_import", "real_env_execution", "observation_structure",
                  "adapter_attachment", "overall"):
            print(f"- {k}: {RESULTS[k]}")
        print("=" * 60, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
