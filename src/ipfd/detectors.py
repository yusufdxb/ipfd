"""Silent-failure detectors.

Three deliberately simple, non-ML detectors, each returning a per-timestep score
in ``[0, 1]`` where higher means "more anomalous relative to this rollout's own
early-episode baseline". They are combined into a single failure-imminence score.

Design choices:
- Every detector is *self-referential*: it calibrates against a baseline window at
  the start of the same rollout. No training set, no stored statistics, no GPU.
- Scores are robust: baseline location/scale use median and MAD, not mean/std, so a
  single outlier step does not poison the baseline.
- A detector for which the required signal is missing (e.g. no entropy) contributes
  a zero score rather than raising, so the combiner degrades gracefully.
"""

from __future__ import annotations

import numpy as np

from .types import Rollout

__all__ = [
    "action_variance_score",
    "entropy_collapse_score",
    "representation_drift",
    "drift_score",
    "failure_imminence_score",
    "first_alarm",
]

_EPS = 1e-9


def _robust_baseline(x: np.ndarray, baseline_window: int) -> tuple[float, float]:
    """Return (location, scale) of the first ``baseline_window`` samples.

    Location is the median; scale is 1.4826 * MAD (a robust std estimate),
    floored to a small positive value so downstream division is safe.
    """
    base = x[: max(1, baseline_window)]
    loc = float(np.median(base))
    mad = float(np.median(np.abs(base - loc)))
    # Use the larger of a robust (MAD) and a classical (std) scale estimate. MAD
    # alone collapses toward zero on very calm baselines, which turns ordinary
    # calm-phase jitter into spurious alarms; std guards against that.
    scale = max(1.4826 * mad, float(np.std(base)))
    return loc, max(scale, _EPS)


def _upper_deviation_score(x: np.ndarray, baseline_window: int, k: float = 3.0) -> np.ndarray:
    """Map how far each sample sits *above* the robust baseline into ``[0, 1]``.

    A sample at ``loc + k*scale`` maps to ~0.5; the logistic saturates beyond.
    """
    loc, scale = _robust_baseline(x, baseline_window)
    z = (x - loc) / scale
    return 1.0 / (1.0 + np.exp(-(z - k)))


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling standard deviation, same length as ``x`` (edge-padded)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}.")
    T = x.shape[0]
    out = np.zeros(T)
    for t in range(T):
        lo = max(0, t - window + 1)
        out[t] = np.std(x[lo : t + 1])
    return out


def action_variance_score(
    actions: np.ndarray, window: int = 5, baseline_window: int = 20
) -> np.ndarray:
    """Detect spikes in short-horizon action variance.

    A policy that has entered a doomed trajectory often starts to thrash: the
    per-step variance of its action vector jumps above its calm-phase baseline.

    Args:
        actions: ``(T, act_dim)`` action array.
        window: Trailing window for the rolling variance.
        baseline_window: Number of leading steps used to calibrate "normal".

    Returns:
        ``(T,)`` score in ``[0, 1]``.
    """
    try:
        actions = np.asarray(actions)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "actions must be a numeric 2-D array with shape (T, act_dim)."
        ) from exc
    if actions.ndim != 2:
        raise ValueError(f"actions must be a 2-D array with shape (T, act_dim), got {actions.shape}.")
    if actions.shape[0] == 0:
        raise ValueError("actions has zero timesteps (T=0); expected at least one timestep.")
    if actions.shape[1] == 0:
        raise ValueError("actions has zero action dimensions (act_dim=0); expected at least one action dimension.")
    try:
        actions = actions.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"actions must contain only numeric values convertible to float64, got dtype {actions.dtype}."
        ) from exc
    if not np.isfinite(actions).all():
        bad = tuple(int(i) for i in np.argwhere(~np.isfinite(actions))[0])
        raise ValueError(f"actions contains a non-finite value at index {list(bad)}.")
    per_dim_std = np.stack([_rolling_std(actions[:, i], window) for i in range(actions.shape[1])], axis=1)
    agg = per_dim_std.mean(axis=1)
    return _upper_deviation_score(agg, baseline_window)


def entropy_collapse_score(
    entropy: np.ndarray | None, baseline_window: int = 20, k: float = 3.0
) -> np.ndarray:
    """Detect abnormal *drops* in policy entropy (overconfidence collapse).

    The value this whole tool exists to surface: a policy can become *more*
    confident (entropy collapsing toward zero) precisely as it commits to a broken
    trajectory. We therefore score how far entropy falls *below* baseline.

    Returns an all-zero score if ``entropy`` is ``None``.
    """
    if entropy is None:
        return np.zeros(0)
    loc, scale = _robust_baseline(entropy, baseline_window)
    z = (loc - entropy) / scale  # positive when entropy drops below baseline
    return 1.0 / (1.0 + np.exp(-(z - k)))


def representation_drift(
    embeddings: np.ndarray | None, ref_window: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """Raw per-step drift of the latent embedding from an early reference.

    Args:
        embeddings: ``(T, emb_dim)`` latents, or ``None``.
        ref_window: Steps averaged to form the reference embedding.
        metric: ``"cosine"`` (1 - cosine similarity) or ``"l2"``.

    Returns:
        ``(T,)`` non-negative drift, or an empty array if ``embeddings`` is ``None``.
    """
    if embeddings is None:
        return np.zeros(0)
    ref = embeddings[: max(1, ref_window)].mean(axis=0)
    if metric == "cosine":
        ref_n = ref / (np.linalg.norm(ref) + _EPS)
        emb_n = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + _EPS)
        return 1.0 - emb_n @ ref_n
    if metric == "l2":
        return np.linalg.norm(embeddings - ref, axis=1)
    raise ValueError(f"unknown metric {metric!r}")


def drift_score(
    embeddings: np.ndarray | None,
    ref_window: int = 10,
    baseline_window: int = 20,
    metric: str = "cosine",
) -> np.ndarray:
    """Representation drift normalized to ``[0, 1]`` against its own baseline."""
    raw = representation_drift(embeddings, ref_window, metric)
    if raw.size == 0:
        return raw
    return _upper_deviation_score(raw, baseline_window)


def failure_imminence_score(
    rollout: Rollout,
    weights: dict[str, float] | None = None,
    baseline_window: int = 20,
) -> np.ndarray:
    """Combine the individual detectors into one imminence score in ``[0, 1]``.

    The combiner is a weighted maximum over available detectors: a failure is
    imminent if *any* channel screams, which matches how an engineer reads these
    plots. Missing channels are simply dropped from the max.
    """
    w = {"action_variance": 1.0, "entropy_collapse": 1.0, "drift": 1.0}
    if weights:
        unknown = set(weights) - set(w)
        if unknown:
            raise ValueError(
                f"unknown detector weight key(s): {sorted(unknown)}; "
                f"valid keys are {sorted(w)}."
            )
        for name, value in weights.items():
            if not np.isfinite(value):
                raise ValueError(f"detector weight {name!r} must be finite, got {value!r}.")
        w.update(weights)

    T = rollout.T
    channels: list[np.ndarray] = []

    av = action_variance_score(rollout.actions, baseline_window=baseline_window)
    channels.append(np.clip(w["action_variance"], 0, 1) * av)

    ent = entropy_collapse_score(rollout.entropy, baseline_window=baseline_window)
    if ent.size == T:
        channels.append(np.clip(w["entropy_collapse"], 0, 1) * ent)

    dr = drift_score(rollout.embeddings, baseline_window=baseline_window)
    if dr.size == T:
        channels.append(np.clip(w["drift"], 0, 1) * dr)

    return np.max(np.stack(channels, axis=1), axis=1)


def first_alarm(score: np.ndarray, threshold: float = 0.5, persistence: int = 3) -> int | None:
    """First timestep where ``score`` stays ``>= threshold`` for ``persistence`` steps.

    Persistence suppresses single-step blips so the alarm time is meaningful as a
    "the tool would have paged you here" moment.
    """
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence!r}.")
    if score.size == 0:
        return None
    hot = score >= threshold
    run = 0
    for t in range(score.size):
        run = run + 1 if hot[t] else 0
        if run >= persistence:
            return t - persistence + 1
    return None
