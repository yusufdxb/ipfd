"""Strict Isaac Lab runtime-compatibility validation for IPFD.

Answers one question with evidence, not assumptions:

    Does IPFD operate on REAL Isaac Lab rollouts, or only on synthetic/mock inputs?

Everything the IPFD adapter assumes about the env API (reset/step signatures,
observation dict structure, success/termination signals, and sim state
save/restore for the recovery probe) is *detected at runtime* against a live
Isaac Lab environment and reported. Nothing is mocked.

Run (real Isaac Lab install + GPU required):

    OMNI_KIT_ACCEPT_EULA=YES \\
    ~/Sim/isaac-sim-venv/bin/python scripts/verify_isaac_runtime.py --headless

The final block printed is machine-readable (IPFD_RUNTIME_COMPATIBILITY).
"""

from __future__ import annotations

# ruff: noqa: I001
# Import order is load-bearing: AppLauncher must launch the sim BEFORE any
# isaaclab.* / isaaclab_tasks import, so isort reordering is disabled here.

import argparse
import os
import sys
import traceback

# --- Isaac Lab app must be launched BEFORE importing isaaclab.* submodules ---
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="IPFD Isaac Lab runtime validation")
parser.add_argument("--env_id", default="Isaac-Lift-Cube-Franka-v0",
                    help="Franka single-object manipulation env (closest to pick-and-place).")
parser.add_argument("--steps", type=int, default=48, help="Primary rollout length.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # this is a validation run, never interactive

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- now the sim is up: safe to import the rest ---
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  (registers the Isaac-* gym envs)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# IPFD is under src/ (src-layout). Use the real package, no mocks.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
from ipfd import build_report  # noqa: E402
from ipfd.adapters import isaac_lab as ipfd_adapter  # noqa: E402

# Verdict accumulator -------------------------------------------------------
RESULTS = {
    "real_isaac_lab_execution": "FAIL",
    "ipfd_attachment_valid": "FAIL",
    "runtime_api_compatibility": "FAIL",
    "failure_detection_on_real_env": "FAIL",
    "synthetic_fallback_detected": "YES",  # guilty until proven innocent
    "overall_verdict": "MOCK_DEPENDENT",
}
NOTES: list[str] = []


def log(section: str, msg: str) -> None:
    print(f"[{section}] {msg}", flush=True)


class RandomPolicy:
    """Untrained policy: exercises the API surface, not task competence.

    Returns (action, confidence_proxy, embedding). The confidence and embedding
    are side channels IPFD instruments; with a random policy they carry no
    meaning, which is exactly why we do NOT claim real failure *prediction* here,
    only API compatibility.
    """

    def __init__(self, act_dim: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.act_dim = act_dim

    def act(self, obs: np.ndarray):
        a = 0.1 * self.rng.standard_normal(self.act_dim)
        ent = float(np.linalg.norm(a))            # placeholder confidence proxy
        emb = np.asarray(obs, dtype=float)[:8]    # placeholder latent
        return a, ent, emb


def main() -> None:
    env = None
    try:
        # === 1. Real Isaac Lab execution test ==============================
        log("1", f"creating live env: {args.env_id} (num_envs=1, device={args.device})")
        env_cfg = parse_env_cfg(args.env_id, device=args.device, num_envs=1)
        env = gym.make(args.env_id, cfg=env_cfg)
        log("1", f"env class: {type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}")
        # confirm this is genuinely Isaac Lab code, not a stub
        real_module = type(env.unwrapped).__module__.startswith("isaaclab")

        obs, info = env.reset(seed=args.seed)
        obs_is_dict = isinstance(obs, dict)
        log("1", f"reset() -> obs type={type(obs).__name__}, info keys={list(info.keys()) if isinstance(info, dict) else info}")
        if obs_is_dict:
            for k, v in obs.items():
                shape = tuple(v.shape) if hasattr(v, "shape") else "?"
                log("1", f"  obs['{k}'] shape={shape} dtype={getattr(v,'dtype','?')}")

        act_dim = int(env.action_space.shape[-1])
        log("1", f"action_space={env.action_space.shape} -> act_dim={act_dim}")

        term_seen = trunc_seen = False
        reward_seen = False
        for _i in range(5):
            a = torch.as_tensor(0.1 * np.random.randn(1, act_dim), dtype=torch.float32, device=args.device)
            step_out = env.step(a)
            assert len(step_out) == 5, f"env.step returned {len(step_out)}-tuple, expected 5"
            o, rew, terminated, truncated, inf = step_out
            reward_seen = reward_seen or (rew is not None)
            term_seen = term_seen or bool(torch.as_tensor(terminated).any().item())
            trunc_seen = trunc_seen or bool(torch.as_tensor(truncated).any().item())
        log("1", f"5x step(): 5-tuple OK, reward_present={reward_seen}, "
                  f"any_terminated={term_seen}, any_truncated={trunc_seen}")

        if real_module and obs_is_dict and reward_seen:
            RESULTS["real_isaac_lab_execution"] = "PASS"
        NOTES.append(f"env module real={real_module}, obs_dict={obs_is_dict}")

        # === 3. Runtime API compatibility (what the adapter assumes) =======
        default_obs_key = "policy"
        obs_key_ok = (not obs_is_dict) or (default_obs_key in obs)
        log("3", f"adapter default obs_key='{default_obs_key}' present in obs dict: {obs_key_ok}")

        # _extract_obs must yield a flat float64 vector from the live obs
        try:
            vec = ipfd_adapter._extract_obs(obs, default_obs_key)
            extract_ok = isinstance(vec, np.ndarray) and vec.ndim == 1 and vec.dtype == np.float64
            log("3", f"_extract_obs(live_obs) -> shape={vec.shape} dtype={vec.dtype} ok={extract_ok}")
        except Exception as e:
            extract_ok = False
            log("3", f"_extract_obs FAILED on live obs: {e!r}")

        # sim state save/restore API (the critical recovery-probe touchpoint)
        scene = getattr(env.unwrapped, "scene", None)
        has_get_state = hasattr(scene, "get_state")
        has_reset_to = hasattr(scene, "reset_to")
        state_methods = sorted(m for m in dir(scene) if ("state" in m.lower() or "reset" in m.lower())
                               and not m.startswith("__")) if scene is not None else []
        log("3", f"scene.get_state={has_get_state}  scene.reset_to={has_reset_to}")
        log("3", f"scene state/reset-ish methods: {state_methods}")

        if obs_key_ok and extract_ok:
            RESULTS["runtime_api_compatibility"] = "PASS"

        # === 2. IPFD integration on the LIVE rollout ======================
        # Reuse the untrained policy as its own loose recovery controller ONLY
        # if the save/restore API exists; otherwise skip the probe (PoNR=None).
        policy = RandomPolicy(act_dim, args.seed)
        recovery = policy if (has_get_state and has_reset_to) else None
        log("2", f"collect_rollout via ipfd.adapters.isaac_lab (recovery_probe={'on' if recovery else 'off'})")
        rollout = ipfd_adapter.collect_rollout(
            env, policy, seed=args.seed, max_steps=args.steps,
            recovery_controller=recovery, recovery_stride=8, recovery_budget=12,
        )
        log("2", f"Rollout: T={rollout.T} obs={rollout.observations.shape} act={rollout.actions.shape} "
                  f"entropy={'yes' if rollout.entropy is not None else 'no'} "
                  f"emb={'yes' if rollout.embeddings is not None else 'no'} "
                  f"success={rollout.success} t_failure={rollout.t_failure} "
                  f"recovery={'yes' if rollout.recovery_success is not None else 'no'}")
        log("2", f"rollout.meta['source'] = {rollout.meta.get('source')!r}")

        report = build_report(rollout)
        log("2", "build_report() on LIVE rollout succeeded:")
        for line in report.summary().splitlines():
            log("2", "  " + line)

        rollout_is_real = (
            rollout.meta.get("source") == "isaac_lab"
            and rollout.T > 0
            and rollout.observations.shape[0] == rollout.T
        )
        if rollout_is_real and report is not None:
            RESULTS["ipfd_attachment_valid"] = "PASS"
            RESULTS["synthetic_fallback_detected"] = "NO"

        # === 4. Failure handling on the real env ==========================
        # A random policy cannot lift the cube, so the episode ends in
        # failure (termination/truncation). We check IPFD registers the
        # observable failure. PoNR/progression is only meaningful when a real
        # recovery oracle exists AND the save/restore API is present.
        observed_failure = (not rollout.success) and (rollout.t_failure is not None)
        probe_ran = rollout.recovery_success is not None
        # Degenerate PoNR: an UNTRAINED recovery oracle never recovers, so
        # recovery_success is all-False and PoNR collapses to step 0. The probe
        # PLUMBING is real (state save/restore executed), but the VALUE is not a
        # meaningful point of no return. Be honest about that distinction.
        all_false_recovery = probe_ran and (not bool(np.any(rollout.recovery_success)))
        degenerate_ponr = (report.t_ponr == 0) or all_false_recovery
        meaningful_alarm = report.t_alarm is not None
        log("4", f"observable failure registered: {observed_failure} (t_failure={rollout.t_failure})")
        log("4", f"recovery probe executed (state save/restore): {probe_ran}")
        log("4", f"PoNR value={report.t_ponr}  degenerate(untrained oracle)={degenerate_ponr}  "
                  f"detector_alarm={'fired' if meaningful_alarm else 'n/a'}")
        if observed_failure and meaningful_alarm and not degenerate_ponr:
            # Only reachable with a trained policy + real recovery controller.
            RESULTS["failure_detection_on_real_env"] = "PASS"
        elif observed_failure and probe_ran:
            RESULTS["failure_detection_on_real_env"] = "PARTIAL"
            NOTES.append("Failure-detection PIPELINE ran on real data end to end (observable "
                         "failure registered; recovery probe executed real state save/restore). "
                         "But this run used an UNTRAINED random policy as its own recovery oracle, "
                         "so recovery_success is all-False and PoNR degenerates to step 0, and the "
                         "detectors have no meaningful signal. Post-failure detection is real; "
                         "meaningful PRE-failure PoNR/imminence needs a trained policy + a real "
                         "recovery controller. That is a policy/oracle gap, NOT an API gap.")
        elif observed_failure:
            RESULTS["failure_detection_on_real_env"] = "PARTIAL"
            NOTES.append("Observable failure registered but recovery probe did not run "
                         f"(get_state={has_get_state}, reset_to={has_reset_to}).")

        # === overall verdict ==============================================
        # overall_verdict answers the posed question: is the RUNTIME INTEGRATION
        # real, or dependent on synthetic/mock assumptions? That rides on the four
        # integration axes below. failure_detection_on_real_env is reported
        # separately and, when PARTIAL only because of an untrained oracle, does
        # NOT demote the integration verdict (it is a policy gap, not an API gap).
        core_real = (RESULTS["real_isaac_lab_execution"] == "PASS"
                     and RESULTS["ipfd_attachment_valid"] == "PASS"
                     and RESULTS["runtime_api_compatibility"] == "PASS"
                     and RESULTS["synthetic_fallback_detected"] == "NO")
        if core_real:
            RESULTS["overall_verdict"] = "REAL_COMPATIBLE"
        elif RESULTS["real_isaac_lab_execution"] == "PASS":
            RESULTS["overall_verdict"] = "PARTIALLY_COMPATIBLE"
        else:
            RESULTS["overall_verdict"] = "MOCK_DEPENDENT"

    except Exception:
        log("ERROR", "verification aborted with exception:")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # emit the machine-readable verdict no matter what
        print("\n" + "=" * 60)
        if NOTES:
            print("NOTES:")
            for n in NOTES:
                print(f"  - {n}")
            print("-" * 60)
        print("IPFD_RUNTIME_COMPATIBILITY:")
        print(f"- real_isaac_lab_execution: {RESULTS['real_isaac_lab_execution']}")
        print(f"- ipfd_attachment_valid: {RESULTS['ipfd_attachment_valid']}")
        print(f"- runtime_api_compatibility: {RESULTS['runtime_api_compatibility']}")
        print(f"- failure_detection_on_real_env: {RESULTS['failure_detection_on_real_env']}")
        print(f"- synthetic_fallback_detected: {RESULTS['synthetic_fallback_detected']}")
        print(f"- overall_verdict: {RESULTS['overall_verdict']}")
        print("=" * 60, flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
