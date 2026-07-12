"""Isaac Lab adapter: collect a Franka pick-and-place rollout with a recovery probe.

STATUS: **runtime-verified on Isaac Lab 4.5.22** (env ``Isaac-Lift-Cube-Franka-IK-Abs-v0``,
a CUDA GPU) via the ``scripts/verify_pnor_*.py`` evidence chain. The recovery
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
    "evaluate_recovery_isolated",
]


class Policy(Protocol):
    """Batched policy interface the adapter drives.

    Called on the wrapped env's observation and returns a batched action tensor
    (one row per env). It may optionally expose per-env ``last_entropy`` and
    ``last_embedding`` attributes -- the side channels IPFD's detectors instrument;
    absent means that detector is disabled -- and a ``reset(dones)`` method.
    """

    def __call__(self, obs: Any) -> Any:
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


def _extract_obs(obs: Any, obs_key: str, env_idx: int = 0) -> np.ndarray:
    """Flatten one env's observation group to a float64 vector (runtime API check)."""
    group = obs[obs_key] if isinstance(obs, dict) else obs
    arr = group.detach().cpu().numpy() if hasattr(group, "detach") else np.asarray(group)
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr[env_idx]
    return arr.reshape(-1).astype(np.float64)


def probe_recovery_isolated(
    env: Any,
    saved_state: Any,
    recovery_policy: Any,
    *,
    recovered: Any,
    primary_env: int = 0,
    probe_env: int = 1,
    budget: int = 90,
    locality: dict[str, float] | None = None,
) -> bool:  # pragma: no cover - requires live sim
    """One env-isolated recovery probe (the VERIFIED mechanic).

    Origin-shift ``saved_state`` into ``probe_env``, ``reset_to`` **only** that env,
    run the batched ``recovery_policy`` there for ``budget`` steps, and report
    whether ``recovered(env, probe_env)`` ever became true. The primary env is never
    restored, so its rollout is untouched.

    Args:
        recovery_policy: Batched callable ``policy(obs) -> actions`` (all envs).
        recovered: ``callable(env, probe_env) -> bool`` success test (e.g. a lift).
        locality: If given (a ``{"max", "n"}`` dict), the probe records the primary
            object pose before and after ``reset_to`` and accumulates the max delta;
            a correct isolation keeps it at ~0. Requires ``num_envs >= 2``.
    """
    import torch
    import warp as wp

    scene = env.unwrapped.scene
    delta = (scene.env_origins[probe_env] - scene.env_origins[primary_env]).detach()
    state_probe = offset_root_positions(saved_state, delta)

    pose_before = (
        wp.to_torch(scene["object"].data.root_pose_w)[primary_env].detach().clone()
        if locality is not None else None
    )
    env_ids = torch.tensor([probe_env], device=env.unwrapped.device, dtype=torch.long)
    scene.reset_to(state_probe, env_ids)
    if locality is not None:
        pose_after = wp.to_torch(scene["object"].data.root_pose_w)[primary_env].detach()
        locality["max"] = max(locality["max"], float((pose_after - pose_before).abs().max().item()))
        locality["n"] = locality.get("n", 0) + 1

    if hasattr(env.unwrapped, "episode_length_buf"):
        env.unwrapped.episode_length_buf[:] = 0  # avoid a timeout auto-reset mid-probe

    obs = env.get_observations()
    for _ in range(budget):
        actions = recovery_policy(obs)
        obs, _reward, dones, _info = env.step(actions)
        if recovered(env, probe_env):
            return True
        if bool(dones[probe_env].item()):
            return False
    return False


def evaluate_recovery_isolated(
    env: Any,
    states: dict[int, Any],
    recovery_policy: Any,
    *,
    recovered: Any,
    primary_env: int = 0,
    probe_env: int = 1,
    budget: int = 90,
) -> tuple[dict[int, bool], dict[str, float]]:  # pragma: no cover - requires live sim
    """Evaluate recovery from each saved checkpoint via env isolation (Pass 2).

    Returns ``(verdicts, locality)``: ``verdicts[step]`` is whether recovery
    succeeded from the state saved at ``step``, and ``locality`` reports the live
    primary-integrity assertion (``max`` env-0 pose delta, ``n`` resets).
    """
    locality = {"max": 0.0, "n": 0}
    verdicts: dict[int, bool] = {}
    for step, state in sorted(states.items()):
        verdicts[step] = probe_recovery_isolated(
            env, state, recovery_policy, recovered=recovered,
            primary_env=primary_env, probe_env=probe_env, budget=budget, locality=locality,
        )
    return verdicts, locality


def collect_rollout(
    env: Any,
    policy: Any,
    *,
    object_height: Any,
    rest_height: float,
    lift_threshold: float = 0.06,
    recovery_policy: Any | None = None,
    primary_env: int = 0,
    probe_env: int = 1,
    max_steps: int = 220,
    probe_stride: int = 8,
    probe_budget: int = 90,
    on_step: Any | None = None,
    obs_key: str = "policy",
    seed: int | None = None,
    meta: dict | None = None,
) -> Rollout:  # pragma: no cover - requires live sim
    """Roll out a batched ``policy`` in ``primary_env`` and build a :class:`Rollout`.

    The packaged implementation of the env-isolated recovery-probe design verified
    end-to-end on a trained rsl_rl policy in ``scripts/verify_learned_policy.py``
    (Isaac Lab 4.5.22). It is a **decoupled two-pass**: record the primary rollout in
    ``primary_env`` (never ``reset_to``), then, if ``recovery_policy`` is given,
    evaluate recovery from the saved checkpoints in ``probe_env`` via
    :func:`evaluate_recovery_isolated`. The single-env interleaved probe -- proven to
    corrupt the primary after a grasp -- is gone.

    Args:
        env: Isaac Lab manager-based RL env, wrapped so ``get_observations()`` and
            ``step(actions)`` take/return batched tensors. ``num_envs >= 2`` when
            ``recovery_policy`` is set.
        policy: Batched callable ``policy(obs) -> actions``. May expose per-env
            ``last_entropy`` / ``last_embedding`` (to feed the detectors) and
            ``reset(dones)``; a missing signal simply disables its detector.
        object_height: ``callable(env, env_idx) -> float`` for the tracked object.
        rest_height: Settled resting height of the object [m] (caller-measured).
        lift_threshold: Rise above ``rest_height`` [m] counted as a lift / recovery.
        recovery_policy: Batched recovery oracle for the probe; ``None`` skips PoNR.
        on_step: Optional ``callable(step, env, actions)`` invoked after the policy
            acts and before ``env.step`` -- the hook a caller uses to inject a fault
            (edit ``actions`` in place, or write the sim). The recorded action is the
            policy's *pre-injection* action, so the detectors see the true policy.

    Returns:
        A fully populated :class:`Rollout` (with ``recovery_success`` when probed).
    """
    unwrapped = getattr(env, "unwrapped", env)
    n_envs = int(getattr(unwrapped, "num_envs", 1))
    if recovery_policy is not None and n_envs < 2:
        raise ValueError(
            "The recovery probe requires num_envs >= 2 (environment isolation): "
            "env 0 is the pristine primary and env 1 is the probe cell. A single-env "
            "reset_to probe poisons the sim (see verify_pnor_decoupled.py). Create the "
            "env with num_envs>=2, or omit recovery_policy to skip PoNR."
        )

    _require_isaac_lab()

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    ent_list: list[float] = []
    emb_list: list[np.ndarray] = []

    obs = env.get_observations()
    states: dict[int, Any] = {}
    t_failure: int | None = None

    for step in range(max_steps):
        if recovery_policy is not None and step % probe_stride == 0:
            states[step] = slice_state(
                unwrapped.scene.get_state(), slice(primary_env, primary_env + 1)
            )

        actions = policy(obs)
        obs_list.append(np.asarray(obs[obs_key][primary_env].detach().cpu(), dtype=np.float64).reshape(-1))
        act_list.append(np.asarray(actions[primary_env].detach().cpu(), dtype=np.float64).reshape(-1))
        ent = getattr(policy, "last_entropy", None)
        emb = getattr(policy, "last_embedding", None)
        if ent is not None:
            ent_list.append(float(ent[primary_env]))
        if emb is not None:
            emb_list.append(np.asarray(emb[primary_env], dtype=np.float64).reshape(-1))

        if on_step is not None:
            on_step(step, env, actions)

        obs, _reward, dones, _info = env.step(actions)
        if hasattr(policy, "reset"):
            policy.reset(dones)
        if bool(dones[primary_env].item()):
            t_failure = step
            break

    T = len(obs_list)
    success = False
    if t_failure is None:
        success = (object_height(env, primary_env) - rest_height) > lift_threshold
        if not success:
            t_failure = T - 1

    recovery = None
    locality = {"max": 0.0, "n": 0}
    if recovery_policy is not None and states:
        env.reset()  # clean base before probing; probe reset_to overrides probe_env

        def recovered(e: Any, i: int) -> bool:
            return (object_height(e, i) - rest_height) > lift_threshold

        verdicts, locality = evaluate_recovery_isolated(
            env, states, recovery_policy, recovered=recovered,
            primary_env=primary_env, probe_env=probe_env, budget=probe_budget,
        )
        recovery = forward_fill_recovery(verdicts, T)

    full_meta = {
        "source": "isaac_lab",
        "robot": "franka",
        "task": "pick_place",
        "probe": "env_isolated" if recovery_policy is not None else "none",
        "primary_integrity_max_delta": locality["max"],
        "probe_resets": locality["n"],
    }
    if meta:
        full_meta.update(meta)

    return Rollout(
        observations=np.asarray(obs_list),
        actions=np.asarray(act_list),
        entropy=np.asarray(ent_list) if len(ent_list) == T and ent_list else None,
        embeddings=np.asarray(emb_list) if len(emb_list) == T and emb_list else None,
        success=success,
        t_failure=None if success else t_failure,
        recovery_success=recovery,
        dt=float(getattr(unwrapped, "step_dt", 1.0 / 60.0)),
        seed=seed,
        meta=full_meta,
    )
