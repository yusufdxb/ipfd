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


def _reject_nonfinite(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if a floating array holds any NaN or Inf.

    The message names the field, the offending value, and its index so the caller
    knows exactly what is wrong and where. Non-floating arrays cannot hold NaN/Inf
    and are passed through untouched.
    """
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating)):
        return
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    val = arr[idx]
    raise ValueError(
        f"{name} contains a non-finite value ({val}) at index {list(idx)}; "
        f"{name} must be all finite (no NaN or Inf)."
    )


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
        # A rollout must contain at least one timestep. (A negative length is not
        # representable by a NumPy array shape, so T < 0 cannot occur here.)
        if T == 0:
            raise ValueError(
                "observations has zero timesteps (T=0); a Rollout must contain at least one timestep."
            )
        if self.observations.shape[1] == 0:
            raise ValueError("observations has zero feature dimensions (obs_dim=0); expected at least one feature.")
        if self.actions.shape[1] == 0:
            raise ValueError("actions has zero action dimensions (act_dim=0); expected at least one action dimension.")
        if self.actions.shape[0] != T:
            raise ValueError(f"actions length {self.actions.shape[0]} != observations length {T}")
        _reject_nonfinite(self.observations, "observations")
        _reject_nonfinite(self.actions, "actions")

        if self.entropy is not None:
            self.entropy = np.asarray(self.entropy, dtype=np.float64).reshape(-1)
            if self.entropy.shape[0] != T:
                raise ValueError(f"entropy length {self.entropy.shape[0]} != T {T}")
            _reject_nonfinite(self.entropy, "entropy")
        if self.embeddings is not None:
            self.embeddings = np.asarray(self.embeddings, dtype=np.float64)
            if self.embeddings.ndim != 2 or self.embeddings.shape[0] != T:
                raise ValueError(f"embeddings must be (T, emb_dim) with T={T}, got {self.embeddings.shape}")
            _reject_nonfinite(self.embeddings, "embeddings")
        if self.recovery_success is not None:
            # Check for NaN/Inf *before* the bool cast, which would silently turn a
            # NaN into True and hide the corruption.
            _reject_nonfinite(np.asarray(self.recovery_success), "recovery_success")
            self.recovery_success = np.asarray(self.recovery_success, dtype=bool).reshape(-1)
            if self.recovery_success.shape[0] != T:
                raise ValueError(f"recovery_success length {self.recovery_success.shape[0]} != T {T}")

        if not np.isfinite(self.dt) or not self.dt > 0:
            raise ValueError(
                f"dt must be finite and > 0 seconds, got {self.dt!r}; expected a positive timestep."
            )

        if self.t_failure is not None:
            if self.success:
                raise ValueError("success=True is inconsistent with t_failure being set; omit t_failure on success.")
            # bool is a subclass of int in Python, so it must be rejected explicitly
            # before the int check below, or True/False would sneak through as 1/0.
            if isinstance(self.t_failure, (bool, np.bool_)):
                raise ValueError(
                    f"t_failure must be an integer timestep index, got bool {self.t_failure!r}; "
                    f"expected an int in [0, {T})."
                )
            if isinstance(self.t_failure, (float, np.floating)):
                raise ValueError(
                    f"t_failure must be an integer timestep index, got float {self.t_failure!r}; "
                    f"expected an int in [0, {T})."
                )
            if not isinstance(self.t_failure, (int, np.integer)):
                raise ValueError(
                    f"t_failure must be an integer timestep index, got "
                    f"{type(self.t_failure).__name__} {self.t_failure!r}; expected an int in [0, {T})."
                )
            if not (0 <= self.t_failure < T):
                raise ValueError(
                    f"t_failure {self.t_failure} out of range [0, {T}); expected 0 <= t_failure < T."
                )

    @property
    def T(self) -> int:
        """Number of timesteps in the rollout."""
        return self.observations.shape[0]

    def time_s(self) -> np.ndarray:
        """Per-step time axis in seconds."""
        return np.arange(self.T, dtype=np.float64) * self.dt
