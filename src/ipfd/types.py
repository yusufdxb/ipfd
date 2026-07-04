"""Core data types for IPFD.

A :class:`Rollout` is the single unit of analysis: a fixed-length, time-ordered
record of one episode of a policy acting in an environment. Every field is a
plain NumPy array so the entire analysis layer runs without a simulator or GPU.

The simulator adapter (``ipfd.adapters.isaac_lab``) is responsible for *producing*
a Rollout, including the ``recovery_success`` probe array. Everything downstream
(detectors, point-of-no-return, metrics, report, viz) consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Rollout"]


@dataclass
class Rollout:
    """One episode of a policy acting in an environment.

    Args:
        observations: ``(T, obs_dim)`` per-step observation vectors (may be a
            compressed/keyed subset of the full observation).
        actions: ``(T, act_dim)`` per-step actions emitted by the policy.
        entropy: Optional ``(T,)`` policy entropy (or any confidence proxy where
            *lower means more confident*). ``None`` if the policy is deterministic
            and no confidence signal is available.
        embeddings: Optional ``(T, emb_dim)`` latent features (e.g. the penultimate
            layer of the policy). ``None`` if not instrumented.
        success: Whether the episode ended in task success.
        t_failure: Index of the timestep at which failure became *externally
            observable* (e.g. object dropped, timeout). ``None`` on success.
        recovery_success: Optional ``(T,)`` boolean array from the recovery probe:
            ``recovery_success[t]`` is ``True`` if the task is still recoverable
            from the state at step ``t`` under the best-effort recovery controller.
            This is what makes the Point-of-No-Return computable. ``None`` if the
            rollout was collected without a recovery probe.
        dt: Simulation timestep in seconds. Used to report metrics in seconds.
        seed: The seed used to collect the rollout, for reproducibility.
        meta: Free-form metadata (task name, policy id, env config hash, ...).
    """

    observations: np.ndarray
    actions: np.ndarray
    success: bool
    entropy: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    t_failure: int | None = None
    recovery_success: np.ndarray | None = None
    dt: float = 1.0 / 60.0
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float64)
        self.actions = np.asarray(self.actions, dtype=np.float64)
        if self.observations.ndim != 2:
            raise ValueError(f"observations must be (T, obs_dim), got {self.observations.shape}")
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be (T, act_dim), got {self.actions.shape}")
        T = self.observations.shape[0]
        if self.actions.shape[0] != T:
            raise ValueError(f"actions length {self.actions.shape[0]} != observations length {T}")

        if self.entropy is not None:
            self.entropy = np.asarray(self.entropy, dtype=np.float64).reshape(-1)
            if self.entropy.shape[0] != T:
                raise ValueError(f"entropy length {self.entropy.shape[0]} != T {T}")
        if self.embeddings is not None:
            self.embeddings = np.asarray(self.embeddings, dtype=np.float64)
            if self.embeddings.ndim != 2 or self.embeddings.shape[0] != T:
                raise ValueError(f"embeddings must be (T, emb_dim) with T={T}, got {self.embeddings.shape}")
        if self.recovery_success is not None:
            self.recovery_success = np.asarray(self.recovery_success, dtype=bool).reshape(-1)
            if self.recovery_success.shape[0] != T:
                raise ValueError(f"recovery_success length {self.recovery_success.shape[0]} != T {T}")
        if self.t_failure is not None and not (0 <= self.t_failure < T):
            raise ValueError(f"t_failure {self.t_failure} out of range [0, {T})")

    @property
    def T(self) -> int:
        """Number of timesteps in the rollout."""
        return self.observations.shape[0]

    def time_s(self) -> np.ndarray:
        """Per-step time axis in seconds."""
        return np.arange(self.T, dtype=np.float64) * self.dt
