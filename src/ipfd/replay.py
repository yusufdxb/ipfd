"""Record a :class:`Rollout` to disk and replay it without a simulator.

This is the seam that makes IPFD's central claim independently checkable. A
:class:`Rollout` collected from a live Isaac Lab GPU session (the *only* part of
the pipeline that needs a simulator) is a bag of plain NumPy arrays. Persist it
once with :func:`save_rollout`, and anyone can reload it with :func:`load_rollout`
and re-run the full analysis (:func:`ipfd.build_report`) on a CPU, in CI, offline,
forever -- getting the byte-for-byte same report.

Format: a single compressed ``.npz`` holding only NumPy arrays (so it stays small
and portable) plus a short JSON sidecar string for the free-form ``meta`` dict.
Optional fields (``entropy``, ``embeddings``, ``recovery_success``) are simply
absent from the archive when the rollout did not carry them.

    from ipfd.replay import save_rollout, load_rollout
    save_rollout(rollout, "rollout.npz")
    same = load_rollout("rollout.npz")            # no GPU, no Isaac Lab
    report = build_report(same)                   # identical to the live report
"""

from __future__ import annotations

import json

import numpy as np

from .types import Rollout

__all__ = ["save_rollout", "load_rollout"]

_NONE_INT = -1  # backward-compatible sentinel for archives without presence flags


def save_rollout(rollout: Rollout, path: str) -> None:
    """Write ``rollout`` to a compressed ``.npz`` (NumPy arrays only).

    Scalars are stored as 0-d arrays and ``meta`` as a JSON string, so the archive
    contains nothing but arrays and can be shared as a compact regression fixture.
    """
    arrays: dict[str, np.ndarray] = {
        "observations": np.asarray(rollout.observations, dtype=np.float64),
        "actions": np.asarray(rollout.actions, dtype=np.float64),
        "success": np.array(bool(rollout.success)),
        "t_failure": np.array(_NONE_INT if rollout.t_failure is None else int(rollout.t_failure)),
        "has_t_failure": np.array(rollout.t_failure is not None),
        "dt": np.array(float(rollout.dt)),
        "seed": np.array(_NONE_INT if rollout.seed is None else int(rollout.seed)),
        "has_seed": np.array(rollout.seed is not None),
        # Preserve meta insertion order so save -> load -> report is byte-identical.
        "meta_json": np.array(json.dumps(rollout.meta, default=_json_default)),
    }
    if rollout.entropy is not None:
        arrays["entropy"] = np.asarray(rollout.entropy, dtype=np.float64)
    if rollout.embeddings is not None:
        arrays["embeddings"] = np.asarray(rollout.embeddings, dtype=np.float64)
    if rollout.recovery_success is not None:
        arrays["recovery_success"] = np.asarray(rollout.recovery_success, dtype=bool)
    np.savez_compressed(path, **arrays)


def load_rollout(path: str) -> Rollout:
    """Load a :class:`Rollout` written by :func:`save_rollout`. No simulator needed."""
    with np.load(path, allow_pickle=False) as z:
        t_failure = int(z["t_failure"])
        seed = int(z["seed"])
        has_t_failure = bool(z["has_t_failure"]) if "has_t_failure" in z else t_failure != _NONE_INT
        has_seed = bool(z["has_seed"]) if "has_seed" in z else seed != _NONE_INT
        return Rollout(
            observations=z["observations"],
            actions=z["actions"],
            success=bool(z["success"]),
            entropy=z["entropy"] if "entropy" in z else None,
            embeddings=z["embeddings"] if "embeddings" in z else None,
            t_failure=t_failure if has_t_failure else None,
            recovery_success=z["recovery_success"] if "recovery_success" in z else None,
            dt=float(z["dt"]),
            seed=seed if has_seed else None,
            meta=json.loads(str(z["meta_json"])),
        )


def _json_default(o: object) -> object:
    """Coerce stray NumPy scalars in ``meta`` to plain Python for JSON."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
