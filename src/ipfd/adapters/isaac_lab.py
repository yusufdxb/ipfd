"""Isaac Lab adapter: collect a Franka pick-and-place rollout with a recovery probe.

STATUS: **runtime-verified on Isaac Lab 4.5.22** (env ``Isaac-Lift-Cube-Franka-IK-Abs-v0``,
RTX-class GPU) via the ``scripts/verify_pnor_*.py`` evidence chain. The recovery
probe here uses **environment isolation**, which is the design those scripts
proved correct:

  * Single-step ``reset_to`` is bit-exact (``verify_state_fidelity.py``), but
    probing *in the primary env* corrupts the rollout after a grasp because the
    PhysX contact-manifold / solver warm-start cache is not part of
    ``scene.get_state()`` (``verify_probe_transparency.py``:
    ``root_cause: CONTACT_STATE_NOT_RESTORED``).
  * A single ``reset_to`` on a ``num_envs == 1`` sim poisons it *even across
    ``env.reset()``* (``verify_pnor_decoupled.py``: ``reset_to_poisons_env: YES``).
  * BUT per-env ``reset_to`` is **local**: churning env 1 leaves env 0
    bit-identical (``verify_multienv_isolation.py``: ``isolation_viable: YES``).

So the recovery probe **requires ``num_envs >= 2``**: env 0 is the PRIMARY and is
never ``reset_to``; env 1 (``probe_env``) receives origin-shifted snapshots of the
primary and diverges freely. ``verify_pnor_grasped.py`` demonstrates the full
mechanic end-to-end with ``overall_status: VERIFIED`` and a live primary-integrity
assertion (max env-0 pose delta ``0.00e+00`` across 51 probe ``reset_to`` calls).

The pure, simulator-free pieces of that mechanic (``slice_state``,
``offset_root_positions``, ``forward_fill_recovery``) live here and are unit-tested
in CI. Only the functions marked ``requires live sim`` touch Isaac Lab, and the
whole module is import-gated so the analysis layer (detectors, PoNR, metrics,
report, viz) runs without Isaac Lab or a GPU.

Reference pattern (Isaac Lab manager-based env):
    env.reset() -> (obs_dict, info)
    env.step(action) -> (obs_dict, reward, terminated, truncated, info)
    env.unwrapped.scene.get_state() / .reset_to(state, env_ids)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..types import Rollout

__all__ = [
    "Policy",
    "collect_rollout",
    "slice_state",
    "offset_root_positions",
    "forward_fill_recovery",
    "probe_recovery_isolated",
]


class Policy(Protocol):
    """Minimal policy interface the adapter drives.

    ``act`` maps an observation to an action and, optionally, side-channel signals
    (entropy/confidence and a latent embedding) that IPFD instruments. Returning
    ``None`` for either side channel disables the corresponding detector.
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


# --- Pure state helpers (no simulator; unit-tested in CI) ---------------------


def _deep_clone(x: Any) -> Any:
    """Clone a nested sim-state container without importing torch.

    Handles torch tensors (duck-typed ``.clone()``), NumPy arrays (``.copy()``),
    and nested dict/list/tuple. Everything else is returned as-is.
    """
    if hasattr(x, "clone"):  # torch.Tensor
        return x.clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, dict):
        return {k: _deep_clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_deep_clone(v) for v in x)
    return x


def slice_state(state: Any, idx: Any) -> Any:
    """Index every leaf tensor of a nested sim state by ``idx`` and clone.

    ``idx`` may be an int, a slice, or a tensor of env ids. Non-indexable leaves
    (scalars, strings) pass through unchanged.
    """
    is_tensor = hasattr(state, "clone") and hasattr(state, "__getitem__") and not isinstance(
        state, (dict, list, tuple)
    )
    if is_tensor:
        return state[idx].clone()
    if isinstance(state, dict):
        return {k: slice_state(v, idx) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(slice_state(v, idx) for v in state)
    return state


def offset_root_positions(state: Any, delta: Any) -> Any:
    """Shift every articulation/rigid-object root position by ``delta``.

    ``scene.get_state()`` poses are ABSOLUTE world coordinates, so a checkpoint
    captured in env 0 must be translated into the probe env's own cell (by the
    difference of ``env_origins``) before ``reset_to``, or the two arms collide.
    Joint states and velocities are origin-independent and left untouched.
    """
    s = _deep_clone(state)
    if not isinstance(s, dict):
        return s
    for grp in ("articulation", "rigid_object"):
        for _name, fields in s.get(grp, {}).items():
            if isinstance(fields, dict) and "root_pose" in fields:
                fields["root_pose"][:, :3] += delta
    return s


def forward_fill_recovery(verdicts: dict[int, bool], length: int) -> np.ndarray:
    """Expand strided recovery verdicts to a length-``length`` boolean array.

    ``verdicts`` maps a probed step index to whether recovery succeeded. Steps
    between probes inherit the most recent verdict (recovery status is assumed to
    persist until the next probe contradicts it). Steps before the first probe
    default to ``True`` (recoverable until proven otherwise), matching the
    ``point_of_no_return`` convention that PoNR is the last True->False flip.
    """
    rec = np.zeros(length, dtype=bool)
    last = True
    for t in range(length):
        if t in verdicts:
            last = bool(verdicts[t])
        rec[t] = last
    return rec


# --- Simulator touchpoints (thin; require a live Isaac Lab GPU env) -----------


def _reset(env: Any, seed: int) -> Any:
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def _step(env: Any, action: Any) -> tuple[Any, Any, bool, bool, dict]:
    import torch

    act = torch.as_tensor(np.asarray(action, dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(act)
    return obs, reward, bool(_as_scalar(terminated)), bool(_as_scalar(truncated)), info


def _extract_obs(obs: Any, obs_key: str, env_idx: int = 0) -> np.ndarray:
    # VERIFY: manager-based envs return a dict keyed by observation group.
    group = obs[obs_key] if isinstance(obs, dict) else obs
    arr = group.detach().cpu().numpy() if hasattr(group, "detach") else np.asarray(group)
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr[env_idx]
    return arr.reshape(-1).astype(np.float64)


def _is_success(info: dict, success_key: str) -> bool:
    # VERIFY: success may live in info, in a termination term, or derive from reward.
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


def probe_recovery_isolated(
    env: Any,
    state_primary: Any,
    recovery_controller: Policy,
    *,
    primary_env: int = 0,
    probe_env: int = 1,
    budget: int = 140,
    success_key: str = "success",
    obs_key: str = "policy",
    locality: dict[str, float] | None = None,
) -> bool:  # pragma: no cover - requires live sim
    """Env-isolated recovery probe (the VERIFIED mechanic).

    Export a primary checkpoint into ``probe_env`` (origin-shifted), ``reset_to``
    **only** that env, run ``recovery_controller`` there for ``budget`` steps, and
    report whether it reached success. The primary env is never restored, so its
    rollout is untouched.

    If ``locality`` (a ``{"max": float, "n": int}`` dict) is supplied, the probe
    asserts primary integrity live: it records the primary object pose before and
    after ``reset_to`` and accumulates the max delta. A correct isolation keeps
    this at ~0 (see ``verify_pnor_grasped.py``, which measures ``0.00e+00``).

    Requires ``num_envs >= 2``.
    """
    import torch

    scene = env.unwrapped.scene
    origins = scene.env_origins
    delta = (origins[probe_env] - origins[primary_env]).detach()
    state_probe = offset_root_positions(state_primary, delta)

    obj = scene["object"].data.root_pose_w
    pose_before = obj[primary_env].detach().clone() if locality is not None else None

    env_ids = torch.tensor([probe_env], device=env.unwrapped.device, dtype=torch.long)
    scene.reset_to(state_probe, env_ids)

    if locality is not None:
        pose_after = scene["object"].data.root_pose_w[primary_env].detach()
        locality["max"] = max(locality["max"], float((pose_after - pose_before).abs().max().item()))
        locality["n"] = locality.get("n", 0) + 1

    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0  # avoid a timeout auto-reset mid-probe

    recovered = False
    obs = env.unwrapped._get_observations() if hasattr(env.unwrapped, "_get_observations") else None
    for _ in range(budget):
        obs_vec = _extract_obs(obs, obs_key, probe_env) if obs is not None else np.zeros(1)
        action, _, _ = recovery_controller.act(obs_vec)
        obs, _reward, terminated, truncated, info = _step(env, action)
        if _is_success(info, success_key):
            recovered = True
            break
        if terminated or truncated:
            break
    return recovered


def collect_rollout(
    env: Any,
    policy: Policy,
    *,
    seed: int = 0,
    max_steps: int = 300,
    recovery_controller: Policy | None = None,
    recovery_budget: int = 140,
    recovery_stride: int = 10,
    success_key: str = "success",
    obs_key: str = "policy",
    primary_env: int = 0,
    probe_env: int = 1,
) -> Rollout:  # pragma: no cover - requires live sim
    """Roll out ``policy`` in env ``primary_env`` and build a :class:`Rollout`.

    When ``recovery_controller`` is given, the Point-of-No-Return probe runs via
    **environment isolation**: every ``recovery_stride`` steps the primary state
    is snapshotted and evaluated in ``probe_env`` by
    :func:`probe_recovery_isolated`, which never disturbs the primary. This is the
    design verified by ``scripts/verify_pnor_grasped.py``; the single-env probe it
    replaces was proven to poison the sim and is gone.

    Args:
        env: Isaac Lab manager-based RL env. If ``recovery_controller`` is set,
            ``num_envs >= 2`` is required (env isolation).
        policy: The policy under test (drives ``primary_env``).
        recovery_controller: Best-effort controller for the probe. Reusing
            ``policy`` yields a valid (loose) PoNR upper bound.

    Returns:
        A fully populated :class:`Rollout`.
    """
    unwrapped = getattr(env, "unwrapped", env)
    n_envs = int(getattr(unwrapped, "num_envs", 1))
    if recovery_controller is not None and n_envs < 2:
        raise ValueError(
            "The recovery probe requires num_envs >= 2 (environment isolation): "
            "env 0 is the pristine primary and env 1 is the probe cell. A single-env "
            "reset_to probe poisons the sim (see verify_pnor_decoupled.py). Create the "
            "env with num_envs>=2, or omit recovery_controller to skip PoNR."
        )

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
    verdicts: dict[int, bool] = {}
    locality = {"max": 0.0, "n": 0}

    for step in range(max_steps):
        obs_vec = _extract_obs(obs, obs_key, primary_env)
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

        if recovery_controller is not None and step % recovery_stride == 0:
            state = slice_state(env.unwrapped.scene.get_state(), slice(primary_env, primary_env + 1))
            verdicts[step] = probe_recovery_isolated(
                env, state, recovery_controller,
                primary_env=primary_env, probe_env=probe_env,
                budget=recovery_budget, success_key=success_key, obs_key=obs_key,
                locality=locality,
            )

        obs, _reward, terminated, truncated, info = _step(env, action)
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
    if recovery_controller is not None and verdicts:
        verdicts.setdefault(T - 1, verdicts[max(verdicts)])
        recovery = forward_fill_recovery(verdicts, T)

    return Rollout(
        observations=np.asarray(obs_list),
        actions=np.asarray(act_list),
        entropy=np.asarray(ent_list) if have_entropy and ent_list else None,
        embeddings=np.asarray(emb_list) if have_embedding and emb_list else None,
        success=success,
        t_failure=None if success else t_failure,
        recovery_success=recovery,
        dt=float(getattr(unwrapped, "step_dt", 1.0 / 60.0)),
        seed=seed,
        meta={
            "source": "isaac_lab",
            "robot": "franka",
            "task": "pick_place",
            "probe": "env_isolated" if recovery_controller is not None else "none",
            "primary_integrity_max_delta": locality["max"],
            "probe_resets": locality["n"],
        },
    )
