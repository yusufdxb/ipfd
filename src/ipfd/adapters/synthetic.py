"""Synthetic rollouts.

Synthetic rollouts let the analysis layer, tests, and example plots run with zero
simulator dependency, and they carry a ground-truth silent-failure structure, so
the detectors and metrics can be asserted against a known answer.

``make_silent_failure_rollout`` encodes the exact phenomenon IPFD exists to expose:
at ``t_ponr`` the object is knocked out of the reachable workspace (recovery becomes
impossible), yet the policy stays *confident* (entropy collapses low) and smooth for
a long stretch afterward, only visibly thrashing right before ``t_failure``.
"""

from __future__ import annotations

import numpy as np

from ..types import Rollout

__all__ = ["make_silent_failure_rollout", "make_success_rollout"]


def make_silent_failure_rollout(
    seed: int = 0,
    T: int = 200,
    t_ponr: int = 90,
    t_failure: int = 160,
    obs_dim: int = 8,
    act_dim: int = 7,
    emb_dim: int = 16,
    dt: float = 1.0 / 60.0,
) -> Rollout:
    """A Franka-shaped rollout that fails silently, then visibly.

    Timeline:
      * ``[0, t_ponr)``      : nominal reach/grasp, calm actions, normal entropy,
                               embeddings near reference. Recoverable throughout.
      * ``[t_ponr, ...)``    : object shoved out of workspace -> irrecoverable. The
                               embedding drifts steadily (the world changed) but the
                               policy's entropy *drops* (overconfident) and actions
                               stay smooth: the silent-doom window.
      * ``~t_failure``       : the policy finally destabilizes; action variance spikes
                               and failure becomes externally observable.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    # --- observations: a workspace feature that jumps at PoNR (object leaves) ---
    obs = 0.02 * rng.standard_normal((T, obs_dim))
    obs[:, 0] += np.linspace(0.0, 0.5, T)  # arm progressing toward goal
    obs[t_ponr:, 1] += 1.0  # object-in-workspace flag flips off (step change)

    # --- actions: calm, then thrash ramping up to (and peaking at) observable failure ---
    actions = 0.03 * rng.standard_normal((T, act_dim))
    ramp0 = max(t_failure - 15, t_ponr)
    ramp = np.clip((t - ramp0) / max(1, t_failure - ramp0), 0.0, 1.0)  # 0 -> 1 by t_failure, then held
    thrash = ramp**2
    actions += 0.6 * thrash[:, None] * rng.standard_normal((T, act_dim))

    # --- entropy: normal, then COLLAPSES (overconfident) through the doom window ---
    entropy = 1.0 + 0.05 * rng.standard_normal(T)
    entropy[t_ponr:] -= np.linspace(0.0, 0.7, T - t_ponr)  # confidence rises as it commits

    # --- embeddings: near reference, then steady drift after PoNR ---
    base = rng.standard_normal(emb_dim)
    emb = base[None, :] + 0.02 * rng.standard_normal((T, emb_dim))
    drift_dir = rng.standard_normal(emb_dim)
    drift_dir /= np.linalg.norm(drift_dir)
    emb[t_ponr:] += np.linspace(0.0, 3.0, T - t_ponr)[:, None] * drift_dir[None, :]

    # --- recovery probe ground truth: recoverable up to (not including) PoNR ---
    recovery_success = np.ones(T, dtype=bool)
    recovery_success[t_ponr:] = False

    return Rollout(
        observations=obs,
        actions=actions,
        entropy=entropy,
        embeddings=emb,
        success=False,
        t_failure=t_failure,
        recovery_success=recovery_success,
        dt=dt,
        seed=seed,
        meta={"source": "synthetic", "scenario": "silent_failure", "robot": "franka", "task": "pick_place"},
    )


def make_success_rollout(
    seed: int = 1,
    T: int = 200,
    obs_dim: int = 8,
    act_dim: int = 7,
    emb_dim: int = 16,
    dt: float = 1.0 / 60.0,
) -> Rollout:
    """A nominal successful rollout, the negative control the detectors must not trip."""
    rng = np.random.default_rng(seed)
    obs = 0.02 * rng.standard_normal((T, obs_dim))
    obs[:, 0] += np.linspace(0.0, 0.5, T)
    actions = 0.03 * rng.standard_normal((T, act_dim))
    entropy = 1.0 + 0.05 * rng.standard_normal(T)
    base = rng.standard_normal(emb_dim)
    emb = base[None, :] + 0.02 * rng.standard_normal((T, emb_dim))
    recovery_success = np.ones(T, dtype=bool)
    return Rollout(
        observations=obs,
        actions=actions,
        entropy=entropy,
        embeddings=emb,
        success=True,
        t_failure=None,
        recovery_success=recovery_success,
        dt=dt,
        seed=seed,
        meta={"source": "synthetic", "scenario": "success", "robot": "franka", "task": "pick_place"},
    )
