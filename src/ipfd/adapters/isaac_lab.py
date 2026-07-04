"""Isaac Lab adapter: collect a Franka pick-and-place rollout with a recovery probe.

STATUS: written against the standard Isaac Lab manager-based RL env API, but
**not yet verified on a live Isaac Lab install**. It is import-gated so the rest of
IPFD (and CI) runs without Isaac Lab or a GPU. The two touchpoints most likely to
need adjustment for your Isaac Lab version are flagged inline with ``VERIFY:``:

  1. how a single environment's full simulation state is saved and restored
     (needed for the recovery probe), and
  2. the exact keys of the observation dict and the success/termination signal.

Everything above this layer (detectors, PoNR, metrics, report, viz) is stable and
tested; only this file talks to the simulator.

Reference pattern (Isaac Lab manager-based env):
    env.reset() -> (obs_dict, info)
    env.step(action) -> (obs_dict, reward, terminated, truncated, info)
    env.unwrapped.scene / env.unwrapped.observation_manager / ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from ..types import Rollout

__all__ = ["Policy", "collect_rollout"]


class Policy(Protocol):
    """Minimal policy interface the adapter drives.

    ``act`` maps an observation to an action and, optionally, side-channel signals
    (entropy/confidence and a latent embedding) that IPFD instruments. Returning
    ``None`` for either side channel simply disables the corresponding detector.
    """

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float | None, np.ndarray | None]:
        ...


def _require_isaac_lab() -> None:
    try:
        import isaaclab  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The Isaac Lab adapter requires a working Isaac Lab install and a GPU. "
            "Install Isaac Lab and run this outside CI. The analysis layer "
            "(ipfd.build_report / plot_timeline) does not need it."
        ) from exc


def collect_rollout(
    env: Any,
    policy: Policy,
    *,
    seed: int = 0,
    max_steps: int = 300,
    recovery_controller: Policy | None = None,
    recovery_budget: int = 60,
    recovery_stride: int = 5,
    success_key: str = "success",
    obs_key: str = "policy",
) -> Rollout:
    """Roll out ``policy`` in a single-env Isaac Lab ``env`` and build a Rollout.

    The recovery probe (which produces ``recovery_success`` and thus the Point of No
    Return) works by, every ``recovery_stride`` steps, saving the sim state, running
    ``recovery_controller`` for ``recovery_budget`` steps, checking for success, then
    restoring the state and continuing the primary rollout. If no recovery controller
    is supplied the probe is skipped and PoNR will be ``None``.

    Args:
        env: A single-environment Isaac Lab manager-based RL env (``num_envs == 1``).
        policy: The policy under test.
        recovery_controller: Best-effort controller used by the probe. Reusing
            ``policy`` here yields a valid (if loose) PoNR upper bound.

    Returns:
        A fully populated :class:`Rollout`.
    """
    _require_isaac_lab()

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    ent_list: list[float] = []
    emb_list: list[np.ndarray] = []
    have_entropy = True
    have_embedding = True

    obs = _reset(env, seed)
    t_failure: int | None = None
    success = False
    recovery_flags: list[bool] = []

    for step in range(max_steps):
        obs_vec = _extract_obs(obs, obs_key)
        action, entropy, embedding = policy.act(obs_vec)

        obs_list.append(obs_vec)
        act_list.append(np.asarray(action, dtype=np.float64).reshape(-1))
        if entropy is None:
            have_entropy = False
        else:
            ent_list.append(float(entropy))
        if embedding is None:
            have_embedding = False
        else:
            emb_list.append(np.asarray(embedding, dtype=np.float64).reshape(-1))

        # Recovery probe (optional, strided to keep cost bounded).
        if recovery_controller is not None and step % recovery_stride == 0:
            recovery_flags.append(
                _probe_recovery(env, recovery_controller, recovery_budget, success_key, obs_key)
            )
        elif recovery_controller is not None:
            recovery_flags.append(recovery_flags[-1] if recovery_flags else True)

        obs, terminated, truncated, info = _step(env, action)
        if _is_success(info, success_key):
            success = True
            break
        if terminated or truncated:
            t_failure = step
            break

    T = len(obs_list)
    if t_failure is None and not success:
        t_failure = T - 1  # ran out of steps without success == observable failure

    recovery = None
    if recovery_controller is not None and recovery_flags:
        # Forward-fill strided probe results to length T.
        rec = np.asarray(recovery_flags[:T], dtype=bool)
        if rec.size < T:
            rec = np.concatenate([rec, np.full(T - rec.size, rec[-1])])
        recovery = rec

    return Rollout(
        observations=np.asarray(obs_list),
        actions=np.asarray(act_list),
        entropy=np.asarray(ent_list) if have_entropy and ent_list else None,
        embeddings=np.asarray(emb_list) if have_embedding and emb_list else None,
        success=success,
        t_failure=None if success else t_failure,
        recovery_success=recovery,
        dt=float(getattr(getattr(env, "unwrapped", env), "step_dt", 1.0 / 60.0)),
        seed=seed,
        meta={"source": "isaac_lab", "robot": "franka", "task": "pick_place"},
    )


# --- Simulator touchpoints (thin, isolated, VERIFY on your Isaac Lab version) ---


def _reset(env: Any, seed: int) -> Any:
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def _step(env: Any, action: np.ndarray) -> tuple[Any, bool, bool, dict]:
    import torch

    act = torch.as_tensor(np.asarray(action, dtype=np.float32)).reshape(1, -1)
    obs, _reward, terminated, truncated, info = env.step(act)
    return obs, bool(_as_scalar(terminated)), bool(_as_scalar(truncated)), info


def _extract_obs(obs: Any, obs_key: str) -> np.ndarray:
    # VERIFY: manager-based envs return a dict keyed by observation group.
    group = obs[obs_key] if isinstance(obs, dict) else obs
    arr = group.detach().cpu().numpy() if hasattr(group, "detach") else np.asarray(group)
    return arr.reshape(-1).astype(np.float64)


def _is_success(info: dict, success_key: str) -> bool:
    # VERIFY: success may live in info, in a termination term, or be derived from reward.
    val = info.get(success_key) if isinstance(info, dict) else None
    return bool(_as_scalar(val)) if val is not None else False


def _as_scalar(x: Any) -> Any:
    if x is None:
        return False
    if hasattr(x, "item"):
        try:
            return x.reshape(-1)[0].item()
        except Exception:  # pragma: no cover
            return x.item()
    if isinstance(x, np.ndarray):
        return x.reshape(-1)[0]
    return x


def _save_state(env: Any) -> Any:  # pragma: no cover - requires live sim
    # VERIFY: exact API differs by version. Common paths:
    #   state = env.unwrapped.scene.get_state()          # newer Isaac Lab
    #   or read/write articulation root + joint state tensors manually.
    return env.unwrapped.scene.get_state()


def _restore_state(env: Any, state: Any) -> None:  # pragma: no cover - requires live sim
    env.unwrapped.scene.reset_to(state)


def _probe_recovery(
    env: Any,
    recovery_controller: Policy,
    budget: int,
    success_key: str,
    obs_key: str,
) -> bool:  # pragma: no cover - requires live sim
    """Save state, try to recover for ``budget`` steps, restore state, report success."""
    state = _save_state(env)
    recovered = False
    obs = env.unwrapped._get_observations() if hasattr(env.unwrapped, "_get_observations") else None
    try:
        for _ in range(budget):
            obs_vec = _extract_obs(obs, obs_key) if obs is not None else np.zeros(1)
            action, _, _ = recovery_controller.act(obs_vec)
            obs, terminated, truncated, info = _step(env, action)
            if _is_success(info, success_key):
                recovered = True
                break
            if terminated or truncated:
                break
    finally:
        _restore_state(env, state)
    return recovered


# A trivial recovery controller users can pass to get a first PoNR estimate.
def scripted_reach_recovery(get_object_pose: Callable[[], np.ndarray]) -> Policy:  # pragma: no cover
    """Return a Policy that drives the gripper toward the object (placeholder).

    A real deployment supplies a proper scripted grasp; this documents the shape.
    """

    class _R:
        def act(self, obs: np.ndarray):
            return np.zeros(7, dtype=np.float64), None, None

    return _R()
